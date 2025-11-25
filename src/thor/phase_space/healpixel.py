from typing import Iterable, Optional, Tuple

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import quivr as qv
import ray
from jax import jit, vmap

from adam_core.constants import Constants as c
from adam_core.coordinates import (
    CoordinateCovariances,
    Origin,
    SphericalCoordinates,
)
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.iter import _iterate_chunks
from thor.orbit import TestOrbits

jax.config.update("jax_enable_x64", True)

MU = c.MU


def compute_lon_lat_boundaries(nside: int, pixel: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lon/lat of the HEALPix pixel boundaries (nest=True), in degrees.

    Parameters
    ----------
    nside : int
        HEALPix nside.
    pixel : int
        HEALPix pixel number.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        Tuple of longitude and latitude boundaries of the pixel in degrees.
    """
    lon, lat = hp.vec2ang(
        hp.boundaries(nside, pixel, nest=True).T,
        lonlat=True,
    )
    return lon, lat


def centers_and_halfwidths(bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given bin edges, return centers and halfwidths. Useful for creating bins that are not uniform
    in size.

    Parameters
    ----------
    bin_edges : np.ndarray
        Bin edges.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        Tuple of centers and halfwidths.
    """
    bin_edges = np.asarray(bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    halfwidths = 0.5 * (bin_edges[1:] - bin_edges[:-1])
    return centers, halfwidths


def kepler_to_spherical_velocities(elements: jnp.ndarray, mu: float = MU) -> jnp.ndarray:
    """
    JAX function mapping (rho, e, nu, psi) → (vr, omega_lon_deg, omega_lat_deg).

    Parameters
    ----------
    elements : jnp.ndarray, shape (4,)
        [rho, e, nu, psi], with:
          rho : heliocentric distance in units of distance (e.g. au)
          e   : eccentricity
          nu  : true anomaly in degrees
          psi : angle in local tangent plane in degrees, where:
                    psi = 0   → purely prograde along +lon
                    psi ∈ (-90, +90) → prograde
                    psi ∈ (90, 270)   → retrograde
    mu : float, optional
        Gravitational parameter in units of distance^3 / time^2. Defaults to MU.

    Returns
    -------
    jnp.ndarray: (3,)
        [vr, omega_lon_deg, omega_lat_deg]
          vr             : linear radial speed
          omega_lon_deg  : angular rate in lon (deg/time)
          omega_lat_deg  : angular rate in lat (deg/time)
    """
    rho, e, nu, psi = elements
    nu = jnp.radians(nu)
    psi = jnp.radians(psi)

    cosnu = jnp.cos(nu)
    sinnu = jnp.sin(nu)
    cospsi = jnp.cos(psi)
    sinpsi = jnp.sin(psi)

    A = 1.0 + e * cosnu
    h2 = mu * rho * A
    h = jnp.sqrt(h2)

    v_r = (mu / h) * e * sinnu
    v_t = (mu / h) * A

    vlon_linear = v_t * cospsi
    vlat_linear = v_t * sinpsi

    omega_lon_rad = vlon_linear / rho
    omega_lat_rad = vlat_linear / rho

    omega_lon_deg = jnp.degrees(omega_lon_rad)
    omega_lat_deg = jnp.degrees(omega_lat_rad)

    return jnp.array([v_r, omega_lon_deg, omega_lat_deg])


_kepler_to_spherical_velocities_vmap = jit(
    vmap(
        kepler_to_spherical_velocities,
        in_axes=(0, None),
    )
)

_kepler_to_spherical_velocities_jac_vmap = jit(
    vmap(
        jax.jacrev(kepler_to_spherical_velocities),
        in_axes=(0, None),
    )
)


def create_healpixel_test_orbit_worker(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    pixels: Iterable[int],
    time: Timestamp,
    nside: int = 64,
) -> TestOrbits:
    """
    Create Kepler-consistent test orbits localized to HEALPix pixels
    at given heliocentric distances, using bin edges for ρ, e, ν, ψ.

    Parameters
    ----------
    rho_bin_edges : np.ndarray
        Bin edges in heliocentric distance rho in units of distance (e.g. au).
    e_bin_edges : np.ndarray
        Bin edges in eccentricity e (0 <= e < 1).
    nu_bin_edges : np.ndarray
        Bin edges in true anomaly nu (degrees).
    psi_bin_edges : np.ndarray
        Bin edges in tangential direction psi (degees) for v_t in local tangent plane.
        Convention: psi = 0 is purely prograde along +lon; psi in (-90, +90) corresponds to prograde directions.
    pixels : Iterable[int]
        HEALPix pixels (nest=True) defining (lon, lat) for localization.
    time : Timestamp
        Representative time at which these test orbits are defined.
    nside : int, optional
        HEALPix nside. Defaults to 64.

    Returns
    -------
    TestOrbits
        Quivr table of test orbits in Cartesian/ecliptic coordinates,
        with orbit_id, bundle_id, and propagated covariances.
    """
    rho_centers, rho_hw = centers_and_halfwidths(rho_bin_edges)
    e_centers, e_hw = centers_and_halfwidths(e_bin_edges)
    nu_centers, nu_hw = centers_and_halfwidths(nu_bin_edges)
    psi_centers, psi_hw = centers_and_halfwidths(psi_bin_edges)

    num_rho = len(rho_centers)
    num_e = len(e_centers)
    num_nu = len(nu_centers)
    num_psi = len(psi_centers)

    num_states_per_rho = num_e * num_nu * num_psi
    e_grid, nu_grid, psi_grid = np.meshgrid(
        e_centers,
        nu_centers,
        psi_centers,
        indexing="ij",
    )

    e_flat = e_grid.ravel(order="C")
    nu_flat = nu_grid.ravel(order="C")
    psi_flat = psi_grid.ravel(order="C")

    e_hw_grid, nu_hw_grid, psi_hw_grid = np.meshgrid(
        e_hw,
        nu_hw,
        psi_hw,
        indexing="ij",
    )
    sigma_e_flat = e_hw_grid.ravel(order="C")
    sigma_nu_flat = nu_hw_grid.ravel(order="C")
    sigma_psi_flat = psi_hw_grid.ravel(order="C")

    test_orbits = TestOrbits.empty()

    for pixel in pixels:
        # Central direction of this pixel (deg)
        lon, lat = hp.pix2ang(nside, pixel, nest=True, lonlat=True)

        # Positional footprint → lon/lat uncertainties
        lon_boundaries, lat_boundaries = compute_lon_lat_boundaries(nside, pixel)
        dlon = np.max(np.abs(lon_boundaries - lon))
        dlat = np.max(np.abs(lat_boundaries - lat))

        num_states = num_rho * num_states_per_rho

        states = np.empty((num_states, 6), dtype=float)
        covs = np.zeros((num_states, 6, 6), dtype=float)
        orbit_ids = np.empty(num_states, dtype=object)

        states[:, 1] = lon
        states[:, 2] = lat

        covs[:, 1, 1] = dlon**2
        covs[:, 2, 2] = dlat**2

        for i, rho in enumerate(rho_centers):
            idx_start = i * num_states_per_rho
            idx_end = (i + 1) * num_states_per_rho

            # Radial distance and its half-width
            sigma_rho = rho_hw[i]

            states[idx_start:idx_end, 0] = rho
            covs[idx_start:idx_end, 0, 0] = sigma_rho**2

            N = e_flat.shape[0]

            rho_array = jnp.full((N,), rho)
            params_batch = jnp.stack(
                [
                    rho_array,
                    jnp.asarray(e_flat),
                    jnp.asarray(nu_flat),
                    jnp.asarray(psi_flat),
                ],
                axis=1,
            )  # (N, 4)

            vels = _kepler_to_spherical_velocities_vmap(
                params_batch, c.MU
            )  # (N, 3): [vr, ω_lon_deg, ω_lat_deg]
            Js = _kepler_to_spherical_velocities_jac_vmap(params_batch, c.MU)  # (N, 3, 4)

            states[idx_start:idx_end, 3] = np.array(vels[:, 0])
            states[idx_start:idx_end, 4] = np.array(vels[:, 1])
            states[idx_start:idx_end, 5] = np.array(vels[:, 2])

            sigma_rho_vec = np.full(N, sigma_rho)
            sigma_e_vec = np.array(sigma_e_flat)
            sigma_nu_vec = np.array(sigma_nu_flat)
            sigma_psi_vec = np.array(sigma_psi_flat)

            sigmas_np = np.stack(
                [sigma_rho_vec, sigma_e_vec, sigma_nu_vec, sigma_psi_vec],
                axis=1,
            )  # (N, 4)

            sigmas = jnp.asarray(sigmas_np)
            Sigma_p_diag = sigmas**2  # (N, 4)

            # ----- Propagate Σ_p (diag) → Σ_v per sample: Σ_v = J Σ_p J^T -----
            # Js: (N, 3, 4), Σ_p_diag: (N, 4)
            Sigma_p_diag_exp = jnp.expand_dims(Sigma_p_diag, axis=1)  # (N, 1, 4)

            # Scale each column of J by corresponding σ²
            J_scaled = Js * Sigma_p_diag_exp  # (N, 3, 4)

            # Σ_v = J_scaled @ J^T, per sample
            cov_v = np.array(jnp.einsum("nij,nkj->nik", Js, J_scaled))  # (N, 3, 3)

            covs[idx_start:idx_end, 3:6, 3:6] = cov_v

            orbit_ids[idx_start:idx_end] = np.array(
                [
                    f"{pixel}_r{rho:.3f}_e{e:.3f}_nu{nu:.3f}_psi{psi:.3f}"
                    for (e, nu, psi) in zip(e_flat, nu_flat, psi_flat)
                ]
            )

        num_states = states.shape[0]

        coords = SphericalCoordinates.from_kwargs(
            rho=states[:, 0],
            lon=states[:, 1],
            lat=states[:, 2],
            vrho=states[:, 3],
            vlon=states[:, 4],
            vlat=states[:, 5],
            time=Timestamp.from_kwargs(
                days=pa.repeat(time.days[0], num_states),
                nanos=pa.repeat(time.nanos[0], num_states),
                scale=time.scale,
            ),
            covariance=CoordinateCovariances.from_matrix(covs),
            origin=Origin.from_kwargs(code=pa.repeat("SUN", num_states)),
            frame="ecliptic",
        )

        test_orbits_i = TestOrbits.from_kwargs(
            orbit_id=orbit_ids,
            bundle_id=pa.repeat(f"{pixel}", num_states),
            coordinates=coords.to_cartesian(),
        )

        test_orbits = qv.concatenate([test_orbits, test_orbits_i])

    return test_orbits


create_healpixel_test_orbit_worker_remote = ray.remote(create_healpixel_test_orbit_worker)


def create_healpixel_test_orbits(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    time: Timestamp,
    nside: int = 64,
    pixels: Optional[Iterable[int]] = None,
    chunk_size: int = 100,
    max_processes: int = 10,
) -> TestOrbits:
    """
    Generate test orbits over HEALPix pixels using bin edges for ρ, e, ν, and ψ.

    Parameters
    ----------
    rho_bin_edges : np.ndarray (N_rho)
        Bin edges for heliocentric distance rho in units of distance (e.g. au).
    e_bin_edges : np.ndarray (N_e)
        Bin edges for eccentricity e.
    nu_bin_edges : np.ndarray (N_nu)
        Bin edges for true anomaly nu (degrees).
    psi_bin_edges : np.ndarray (N_psi)
        Bin edges for tangential direction psi (degrees).
        psi = 0 is purely prograde along +lon; psi in (-90, 90) is prograde.
    time : Timestamp
        Time for the test orbits.
    nside : int
        HEALPix nside.
    pixels : array-like or None
        Pixels to process. If None, use all pixels for this nside.
    chunk_size : int
        Number of pixels per worker chunk.
    max_processes : int
        Max number of Ray processes (CPUs) to use.

    Returns
    -------
    TestOrbits (N_rho * N_e * N_nu * N_psi * N_pixels)
        Test orbits generated over the given healpixels.
    """
    if pixels is None:
        pixels = np.arange(hp.nside2npix(nside))

    use_ray = initialize_use_ray(num_cpus=max_processes)

    if use_ray:
        futures = []
        test_orbits = TestOrbits.empty()

        for pixel_chunk in _iterate_chunks(pixels, chunk_size):
            futures.append(
                create_healpixel_test_orbit_worker_remote.remote(
                    rho_bin_edges,
                    e_bin_edges,
                    nu_bin_edges,
                    psi_bin_edges,
                    pixel_chunk,
                    time,
                    nside=nside,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                test_orbits_chunk = ray.get(finished[0])
                test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
                if test_orbits.fragmented():
                    test_orbits = qv.defragment(test_orbits)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            test_orbits_chunk = ray.get(finished[0])
            test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
            if test_orbits.fragmented():
                test_orbits = qv.defragment(test_orbits)

    else:
        test_orbits = TestOrbits.empty()
        for pixel_chunk in _iterate_chunks(pixels, chunk_size):
            test_orbits_i = create_healpixel_test_orbit_worker(
                rho_bin_edges,
                e_bin_edges,
                nu_bin_edges,
                psi_bin_edges,
                pixel_chunk,
                time,
                nside=nside,
            )
            test_orbits = qv.concatenate([test_orbits, test_orbits_i])

    return test_orbits
