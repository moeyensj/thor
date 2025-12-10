import os
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
    CartesianCoordinates,
    CoordinateCovariances,
    Origin,
    OriginCodes,
    SphericalCoordinates,
)
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from adam_core.utils.iter import _iterate_chunks
from adam_core.utils.spice import get_perturber_state
from thor.orbit import TestOrbits

jax.config.update("jax_enable_x64", True)

MU = c.MU
MIN_SIGMA = 1e-12
COVARIANCE_JITTER = 1e-18


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


def kepler_to_spherical_velocities(elements: jnp.ndarray, lat: float, mu: float = MU) -> jnp.ndarray:
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
    lat : float
        Latitude in degrees.
    mu : float, optional
        Gravitational parameter in units of distance^3 / time^2. Defaults to MU.

    Returns
    -------
    jnp.ndarray: (3,)
        [vr, vlon, vlat]
          vr             : radial velocity
          vlon           : velocity in lon (deg/time)
          vlat           : velocity in lat (deg/time)
    """
    rho, e, nu, psi = elements
    nu = jnp.radians(nu)
    psi = jnp.radians(psi)
    beta = jnp.radians(lat)

    cosnu = jnp.cos(nu)
    sinnu = jnp.sin(nu)
    cospsi = jnp.cos(psi)
    sinpsi = jnp.sin(psi)
    cosbeta = jnp.cos(beta)

    a = rho * (1.0 + e * cosnu) / (1.0 - e**2)
    p = a * (1.0 - e**2)
    h = jnp.sqrt(mu * p)

    v_r = (mu / h) * e * sinnu
    v_t = h / rho

    v_t_lon = v_t * cospsi
    v_t_lat = v_t * sinpsi

    eps = 1e-12
    cosbeta_safe = jnp.where(jnp.abs(cosbeta) < eps, eps * jnp.sign(cosbeta), cosbeta)

    omega_lon_rad = v_t_lon / (rho * cosbeta_safe)
    omega_lat_rad = v_t_lat / rho

    vlon_deg = jnp.degrees(omega_lon_rad)
    vlat_deg = jnp.degrees(omega_lat_rad)

    return jnp.array([v_r, vlon_deg, vlat_deg])


_kepler_to_spherical_velocities_vmap = jit(
    vmap(
        kepler_to_spherical_velocities,
        in_axes=(0, None, None),
    )
)

_kepler_to_spherical_velocities_vmap_lat = jit(
    vmap(
        kepler_to_spherical_velocities,
        in_axes=(0, 0, None),
    )
)

_kepler_to_spherical_velocities_jac_vmap = jit(
    vmap(
        jax.jacrev(kepler_to_spherical_velocities),
        in_axes=(0, None, None),
    )
)


def geocentric_to_heliocentric_cartesian(
    elements: jnp.ndarray, r_earth: jnp.ndarray, mu: float = MU
) -> jnp.ndarray:
    """
    JAX function mapping (rho, lon, lat, e, nu, psi) → (x, y, z, vx, vy, vz).

    Parameters
    ----------
    elements : jnp.ndarray, shape (6,)
        [rho_g, lon_g, lat_g, e, nu, psi], with:
          rho_g : geocentric distance in units of distance (e.g. au)
          lon_g : longitude in degrees
          lat_g : latitude in degrees
          e : eccentricity
          nu : true anomaly in degrees
          psi : angle in local tangent plane in degrees, where:
                    psi = 0   → purely prograde along +lon
                    psi ∈ (-90, +90) → prograde
                    psi ∈ (90, 270)   → retrograde
    r_earth : jnp.ndarray, shape (3,)
        [x, y, z] of Earth in Heliocentric Cartesian
    mu : float, optional
        Gravitational parameter in units of distance^3 / time^2. Defaults to MU.

    Returns
    -------
    jnp.ndarray: (6,)
        [x, y, z, vx, vy, vz] heliocentric Cartesian coordinates
    """
    rho, lon, lat, e, nu, psi = elements

    lon_rad = jnp.radians(lon)
    lat_rad = jnp.radians(lat)

    cos_lat = jnp.cos(lat_rad)
    x_g = rho * cos_lat * jnp.cos(lon_rad)
    y_g = rho * cos_lat * jnp.sin(lon_rad)
    z_g = rho * jnp.sin(lat_rad)

    r_h_vec = jnp.array([x_g, y_g, z_g]) + r_earth
    x_h, y_h, z_h = r_h_vec

    r_h = jnp.linalg.norm(r_h_vec)
    lat_h_rad = jnp.arcsin(z_h / r_h)
    lat_h_deg = jnp.degrees(lat_h_rad)
    lon_h_rad = jnp.arctan2(y_h, x_h)

    # Elements for velocity func: [r_h, e, nu, psi]
    # Note: kepler_to_spherical_velocities expects degrees for angles in elements
    elements = jnp.array([r_h, e, nu, psi])
    v_sph = kepler_to_spherical_velocities(elements, lat_h_deg, mu)
    # v_sph is [vr, vlon, vlat] in (au/d, deg/d, deg/d)

    vr, vlon_deg, vlat_deg = v_sph
    vlon_rad = jnp.radians(vlon_deg)
    vlat_rad = jnp.radians(vlat_deg)

    cos_lat_h = jnp.cos(lat_h_rad)
    sin_lat_h = jnp.sin(lat_h_rad)
    cos_lon_h = jnp.cos(lon_h_rad)
    sin_lon_h = jnp.sin(lon_h_rad)

    r_hat = jnp.array([cos_lat_h * cos_lon_h, cos_lat_h * sin_lon_h, sin_lat_h])
    lon_hat = jnp.array([-sin_lon_h, cos_lon_h, 0.0])
    lat_hat = jnp.array([-sin_lat_h * cos_lon_h, -sin_lat_h * sin_lon_h, cos_lat_h])

    v_r_vec = vr * r_hat
    v_lon_vec = (r_h * cos_lat_h * vlon_rad) * lon_hat
    v_lat_vec = (r_h * vlat_rad) * lat_hat

    v_vec = v_r_vec + v_lon_vec + v_lat_vec

    return jnp.concatenate([r_h_vec, v_vec])


_geocentric_to_heliocentric_cartesian_vmap = jit(
    vmap(
        geocentric_to_heliocentric_cartesian,
        in_axes=(0, None, None),
    )
)

_geocentric_to_heliocentric_cartesian_jac_vmap = jit(
    vmap(
        jax.jacrev(geocentric_to_heliocentric_cartesian),
        in_axes=(0, None, None),
    )
)


def create_healpixel_test_orbit_worker(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    pixels: Iterable[int],
    time: Timestamp,
    origin: OriginCodes,
    nside: int = 64,
    out_dir: Optional[str] = None,
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
    origin : OriginCodes
        Origin code for the test orbits.
    nside : int, optional
        HEALPix nside. Defaults to 64.
    out_dir : str, optional
        If provided, write per-pixel chunks of `TestOrbits` parquet files into this
        directory (one file per HEALPix pixel chunk), and return an empty in-memory
        table. If None (default), all test orbits are concatenated and returned in
        memory.

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
    num_states = num_rho * num_states_per_rho

    times = Timestamp.from_kwargs(
        days=pa.repeat(time.days[0], num_states),
        nanos=pa.repeat(time.nanos[0], num_states),
        scale=time.scale,
    )
    origins = Origin.from_OriginCodes(origin, size=num_states)
    mu = origins.mu()[0]

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

    # Always accumulate a chunk worth of pixels in memory; if `out_dir`
    # is provided we write a single file per pixel-chunk and then return
    # an empty table.
    pixels_array = np.asarray(list(pixels), dtype=int)
    test_orbits = TestOrbits.empty()

    for pixel in pixels_array:
        # Central direction of this pixel (deg)
        lon, lat = hp.pix2ang(nside, pixel, nest=True, lonlat=True)

        # Positional footprint → lon/lat uncertainties
        lon_boundaries, lat_boundaries = compute_lon_lat_boundaries(nside, pixel)
        dlon = np.max(np.abs(lon_boundaries - lon))
        dlat = np.max(np.abs(lat_boundaries - lat))

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
                params_batch, lat, mu
            )  # (N, 3): [vr, ω_lon_deg, ω_lat_deg]
            Js = _kepler_to_spherical_velocities_jac_vmap(params_batch, lat, mu)  # (N, 3, 4)

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

        coords = SphericalCoordinates.from_kwargs(
            rho=states[:, 0],
            lon=states[:, 1],
            lat=states[:, 2],
            vrho=states[:, 3],
            vlon=states[:, 4],
            vlat=states[:, 5],
            time=times,
            covariance=CoordinateCovariances.from_matrix(covs),
            origin=origins,
            frame="ecliptic",
        )

        test_orbits_i = TestOrbits.from_kwargs(
            orbit_id=orbit_ids,
            bundle_id=pa.repeat(f"{pixel}", num_states),
            nside=pa.repeat(nside, num_states),
            coordinates=coords.to_cartesian(),
        )

        test_orbits = qv.concatenate([test_orbits, test_orbits_i])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        start_pixel = int(pixels_array.min())
        end_pixel = int(pixels_array.max())
        filepath = os.path.join(
            out_dir,
            f"healpixel_nside{nside}_pixels{start_pixel}_{end_pixel}.parquet",
        )
        test_orbits.to_parquet(filepath)
        return TestOrbits.empty()

    return test_orbits


create_healpixel_test_orbit_worker_remote = ray.remote(create_healpixel_test_orbit_worker)


def create_healpixel_test_orbits(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    time: Timestamp,
    origin: OriginCodes = OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    nside: int = 64,
    pixels: Optional[Iterable[int]] = None,
    chunk_size: int = 100,
    max_processes: int = 10,
    out_dir: Optional[str] = None,
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
    origin: OriginCodes, optional
        Origin code for the test orbits.
    nside : int, optional
        HEALPix nside.
    pixels : array-like or None
        Pixels to process. If None, use all pixels for this nside.
    chunk_size : int
        Number of pixels per worker chunk.
    max_processes : int
        Max number of Ray processes (CPUs) to use.
    out_dir : str, optional
        If provided, directory where per-pixel parquet files will be written.
        Each pixel will be saved as
        ``healpixel_nside{nside}_pixel{pixel}.parquet`` within this directory.
        When ``out_dir`` is not None, the function returns an empty in-memory
        table after writing all partitions to disk.

    Returns
    -------
    TestOrbits (N_rho * N_e * N_nu * N_psi * N_pixels)
        Test orbits generated over the given healpixels (in memory when
        ``out_dir`` is None, otherwise an empty table is returned).
    """
    write_to_disk = out_dir is not None

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
                    origin,
                    nside=nside,
                    out_dir=out_dir,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                test_orbits_chunk = ray.get(finished[0])
                if not write_to_disk:
                    test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
                    if test_orbits.fragmented():
                        test_orbits = qv.defragment(test_orbits)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            test_orbits_chunk = ray.get(finished[0])
            if not write_to_disk:
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
                origin,
                nside=nside,
                out_dir=out_dir,
            )
            if not write_to_disk:
                test_orbits = qv.concatenate([test_orbits, test_orbits_i])

    return test_orbits


def create_geocentric_healpixel_test_orbit_worker(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    pixels: Iterable[int],
    time: Timestamp,
    origin: OriginCodes = OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    nside: int = 64,
    out_dir: Optional[str] = None,
    max_phase_angle: Optional[float] = None,
    min_q: Optional[float] = None,
    max_q: Optional[float] = None,
) -> TestOrbits:
    """
    Create test orbits with positions defined on a Geocentric grid, but velocities
    defined by Heliocentric orbital elements.

    This results in a test orbit distribution that has smaller physical volumes
    near the observer (scaling with geocentric distance^2).

    Parameters
    ----------
    rho_bin_edges : np.ndarray
        Bin edges in geocentric distance (delta) [au].
    e_bin_edges : np.ndarray
        Bin edges in heliocentric/barycentric eccentricity.
    nu_bin_edges : np.ndarray
        Bin edges in heliocentric/barycentric true anomaly [deg].
    psi_bin_edges : np.ndarray
        Bin edges in heliocentric/barycentric tangent angle [deg].
    pixels : Iterable[int]
        Geocentric HEALPix pixels (nest=True).
    time : Timestamp
        Time at which to generate the orbits.
    nside : int
        HEALPix nside.
    out_dir : str, optional
        If provided, write per-pixel chunks of `TestOrbits` parquet files into this
        directory (one file per HEALPix pixel chunk), and return an empty in-memory
        table. If None (default), all test orbits are concatenated and returned in
        memory.
    max_phase_angle : float, optional
        Maximum allowed phase angle (degrees) between the HEALPix center direction
        (geocentric) and the Earth vector (heliocentric). Pixels exceeding this
        limit are skipped.
    min_q : float, optional
        Minimum allowed heliocentric perihelion distance q (au). States with
        perihelion below this value are skipped.
    max_q : float, optional
        Maximum allowed heliocentric perihelion distance q (au). States with
        perihelion above this value are skipped.

    Returns
    -------
    TestOrbits
        Test orbits in Heliocentric Cartesian coordinates.
    """
    # 1. Get Geocenter Position at this time
    # We assume the observer is the Geocenter for the grid definition
    geocenter = get_perturber_state(OriginCodes.EARTH, time, frame="ecliptic", origin=origin)
    r_earth = geocenter.r[0]
    earth_unit = r_earth / np.linalg.norm(r_earth)

    # Grid centers
    rho_centers, rho_hw = centers_and_halfwidths(rho_bin_edges)
    e_centers, e_hw = centers_and_halfwidths(e_bin_edges)
    nu_centers, nu_hw = centers_and_halfwidths(nu_bin_edges)
    psi_centers, psi_hw = centers_and_halfwidths(psi_bin_edges)

    # Prepare meshgrids for the orbital elements (constant for all positions)
    # We'll tile these later
    e_grid, nu_grid, psi_grid = np.meshgrid(e_centers, nu_centers, psi_centers, indexing="ij")
    e_flat = e_grid.ravel()
    nu_flat = nu_grid.ravel()
    psi_flat = psi_grid.ravel()

    num_states_per_pos = len(e_flat)

    test_orbits = TestOrbits.empty()
    pixels_array = np.asarray(list(pixels), dtype=int)

    for pixel in pixels_array:
        # 2. Define Geocentric Position Direction
        # (lon, lat) here are Geocentric Ecliptic (if frame is ecliptic)
        # usually healpixels are on the sphere.
        geo_lon, geo_lat = hp.pix2ang(nside, pixel, nest=True, lonlat=True)

        # Approximate phase angle cut: angle between HEALPix center direction and
        # Earth vector (Sun -> Earth); small angles correspond to opposition.
        if max_phase_angle is not None:
            pixel_dir = np.array(
                [
                    np.cos(np.deg2rad(geo_lat)) * np.cos(np.deg2rad(geo_lon)),
                    np.cos(np.deg2rad(geo_lat)) * np.sin(np.deg2rad(geo_lon)),
                    np.sin(np.deg2rad(geo_lat)),
                ]
            )
            phase_angle = np.degrees(np.arccos(np.clip(np.dot(pixel_dir, earth_unit), -1.0, 1.0)))
            if phase_angle > max_phase_angle:
                continue

        # Positional footprint → lon/lat uncertainties
        lon_boundaries, lat_boundaries = compute_lon_lat_boundaries(nside, pixel)
        dlon = np.max(np.abs(lon_boundaries - geo_lon))
        dlat = np.max(np.abs(lat_boundaries - geo_lat))

        # We need to construct the Geocentric Position vectors for each rho
        # Convert (rho, geo_lon, geo_lat) -> Cartesian (x, y, z)
        # We can use SphericalCoordinates to handle the transform easily

        num_rho = len(rho_centers)

        # Create topocentric spherical coordinates for this pixel
        # We repeat the direction for each rho
        geo_sph = SphericalCoordinates.from_kwargs(
            rho=rho_centers,
            lon=np.full(num_rho, geo_lon),
            lat=np.full(num_rho, geo_lat),
            vrho=np.zeros(num_rho),  # Placeholders
            vlon=np.zeros(num_rho),
            vlat=np.zeros(num_rho),
            time=Timestamp.from_kwargs(
                days=np.full(num_rho, time.days[0]),
                nanos=np.full(num_rho, time.nanos[0]),
                scale=time.scale,
            ),
            origin=Origin.from_kwargs(code=np.full(num_rho, "EARTH")),
            frame="ecliptic",
        )

        # Prepare parameters for JAX
        # Params: [rho_g, lon_g, lat_g, e, nu, psi]
        # We need to tile everything to num_states_per_rho
        # Tiled geocentric coordinates
        rho_tiled = np.repeat(rho_centers, num_states_per_pos)
        geo_lon_tiled = np.full(len(rho_tiled), geo_lon)
        geo_lat_tiled = np.full(len(rho_tiled), geo_lat)

        # Tiled orbital elements
        e_tiled = np.tile(e_flat, num_rho)
        nu_tiled = np.tile(nu_flat, num_rho)
        psi_tiled = np.tile(psi_flat, num_rho)

        params_batch = jnp.stack(
            [
                jnp.asarray(rho_tiled),
                jnp.asarray(geo_lon_tiled),
                jnp.asarray(geo_lat_tiled),
                jnp.asarray(e_tiled),
                jnp.asarray(nu_tiled),
                jnp.asarray(psi_tiled),
            ],
            axis=1,
        )

        # Compute States (Position + Velocity)
        states_helio = _geocentric_to_heliocentric_cartesian_vmap(params_batch, jnp.asarray(r_earth), MU)
        states_helio_np = np.array(states_helio)

        # Perihelion filter (q = a * (1 - e))
        r_norm = np.linalg.norm(states_helio_np[:, :3], axis=1)
        v_norm2 = np.sum(states_helio_np[:, 3:] ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = 1.0 / (2.0 / r_norm - v_norm2 / MU)
        q_vals = a * (1.0 - e_tiled)
        mask = np.isfinite(q_vals)
        if min_q is not None:
            mask &= q_vals >= min_q
        if max_q is not None:
            mask &= q_vals <= max_q
        if not mask.any():
            continue

        # Apply mask to element grids for consistent downstream use
        rho_tiled = rho_tiled[mask]
        geo_lon_tiled = geo_lon_tiled[mask]
        geo_lat_tiled = geo_lat_tiled[mask]
        e_tiled = e_tiled[mask]
        nu_tiled = nu_tiled[mask]
        psi_tiled = psi_tiled[mask]
        params_batch = params_batch[mask]
        states_helio_np = states_helio_np[mask]

        # Compute Covariances
        # Input uncertainties (sigmas)
        # [sigma_rho, sigma_lon, sigma_lat, sigma_e, sigma_nu, sigma_psi]
        sigma_rho_tiled = np.repeat(rho_hw, num_states_per_pos)[mask]
        sigma_lon_tiled = np.full(mask.sum(), dlon)
        sigma_lat_tiled = np.full(mask.sum(), dlat)

        e_hw_grid, nu_hw_grid, psi_hw_grid = np.meshgrid(e_hw, nu_hw, psi_hw, indexing="ij")
        sigma_e_flat = e_hw_grid.ravel()
        sigma_nu_flat = nu_hw_grid.ravel()
        sigma_psi_flat = psi_hw_grid.ravel()

        sigma_e_tiled = np.tile(sigma_e_flat, num_rho)[mask]
        sigma_nu_tiled = np.tile(sigma_nu_flat, num_rho)[mask]
        sigma_psi_tiled = np.tile(sigma_psi_flat, num_rho)[mask]

        sigmas_np = np.stack(
            [
                sigma_rho_tiled,
                sigma_lon_tiled,
                sigma_lat_tiled,
                sigma_e_tiled,
                sigma_nu_tiled,
                sigma_psi_tiled,
            ],
            axis=1,
        )
        sigmas = jnp.asarray(sigmas_np)
        sigmas = jnp.maximum(sigmas, MIN_SIGMA)
        Sigma_p_diag = sigmas**2  # (N, 6)

        # Jacobian J (N, 6, 6)
        Js = _geocentric_to_heliocentric_cartesian_jac_vmap(params_batch, jnp.asarray(r_earth), MU)

        # Propagate Covariance: Sigma_v = J * Sigma_p * J^T
        Sigma_p_diag_exp = jnp.expand_dims(Sigma_p_diag, axis=1)  # (N, 1, 6)
        J_scaled = Js * Sigma_p_diag_exp  # (N, 6, 6)
        covs_helio = np.array(jnp.einsum("nij,nkj->nik", Js, J_scaled))  # (N, 6, 6)
        covs_helio = covs_helio + np.eye(6)[None, :, :] * COVARIANCE_JITTER  # (N, 6, 6)

        # Construct Final Heliocentric States
        final_cart = CartesianCoordinates.from_kwargs(
            x=states_helio_np[:, 0],
            y=states_helio_np[:, 1],
            z=states_helio_np[:, 2],
            vx=states_helio_np[:, 3],
            vy=states_helio_np[:, 4],
            vz=states_helio_np[:, 5],
            time=Timestamp.from_kwargs(
                days=np.repeat(time.days[0], len(states_helio_np)),
                nanos=np.repeat(time.nanos[0], len(states_helio_np)),
                scale=time.scale,
            ),
            covariance=CoordinateCovariances.from_matrix(covs_helio),
            origin=Origin.from_OriginCodes(origin, size=len(states_helio_np)),
            frame="ecliptic",
        )

        # Create IDs
        # You might want to encode the Geocentric grid info in the ID for debugging
        # e.g. pixel + geo_rho + e + nu + psi
        orbit_ids = [
            f"{pixel}_grho{gr:.3f}_e{e:.3f}_nu{nu:.3f}_psi{psi:.3f}"
            for gr, e, nu, psi in zip(
                rho_tiled,
                e_tiled,
                nu_tiled,
                psi_tiled,
            )
        ]

        chunk = TestOrbits.from_kwargs(
            orbit_id=orbit_ids,
            bundle_id=pa.repeat(f"{pixel}", len(final_cart)),
            nside=pa.repeat(nside, len(final_cart)),
            coordinates=final_cart,
        )

        test_orbits = qv.concatenate([test_orbits, chunk])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        start_pixel = int(pixels_array.min())
        end_pixel = int(pixels_array.max())
        filepath = os.path.join(
            out_dir,
            f"healpixel_nside{nside}_pixels{start_pixel}_{end_pixel}.parquet",
        )
        test_orbits.to_parquet(filepath)
        return TestOrbits.empty()

    return test_orbits


create_geocentric_healpixel_test_orbit_worker_remote = ray.remote(
    create_geocentric_healpixel_test_orbit_worker
)


def create_geocentric_healpixel_test_orbits(
    rho_bin_edges: np.ndarray,
    e_bin_edges: np.ndarray,
    nu_bin_edges: np.ndarray,
    psi_bin_edges: np.ndarray,
    time: Timestamp,
    nside: int = 64,
    origin: OriginCodes = OriginCodes.SOLAR_SYSTEM_BARYCENTER,
    pixels: Optional[Iterable[int]] = None,
    chunk_size: int = 100,
    max_processes: int = 10,
    out_dir: Optional[str] = None,
    max_phase_angle: Optional[float] = None,
    min_q: Optional[float] = None,
    max_q: Optional[float] = None,
) -> TestOrbits:
    """
    Generate test orbits over geocentric HEALPix pixels using bin edges for Geocentric ρ,
    and heliocentric/barycentric e, ν, and ψ.

    Parameters
    ----------
    rho_bin_edges : np.ndarray (N_rho)
        Bin edges for geocentric distance rho in units of distance (e.g. au).
    e_bin_edges : np.ndarray (N_e)
        Bin edges for heliocentric/barycentric eccentricity e.
    nu_bin_edges : np.ndarray (N_nu)
        Bin edges for heliocentric/barycentric true anomaly nu (degrees).
    psi_bin_edges : np.ndarray (N_psi)
        Bin edges for heliocentric/barycentric tangential direction psi (degrees).
        psi = 0 is purely prograde along +lon; psi in (-90, 90) is prograde.
    time : Timestamp
        Time for the test orbits.
    nside : int, optional
        HEALPix nside.
    pixels : array-like or None
        Pixels to process. If None, use all pixels for this nside.
    chunk_size : int
        Number of pixels per worker chunk.
    max_processes : int
        Max number of Ray processes (CPUs) to use.
    out_dir : str, optional
        If provided, directory where per-pixel parquet files will be written.
        Each pixel will be saved as
        ``healpixel_nside{nside}_pixel{pixel}.parquet`` within this directory.
        When ``out_dir`` is not None, the function returns an empty in-memory
        table after writing all partitions to disk.
    max_phase_angle : float, optional
        Maximum allowed phase angle (degrees) between the HEALPix center direction
        and the Earth vector (heliocentric). Pixels exceeding this limit are
        skipped.
    min_q : float, optional
        Minimum allowed heliocentric perihelion distance q (au). States with
        perihelion below this value are skipped.
    max_q : float, optional
        Maximum allowed heliocentric perihelion distance q (au). States with
        perihelion above this value are skipped.

    Returns
    -------
    TestOrbits (N_rho * N_e * N_nu * N_psi * N_pixels)
        Test orbits generated over the given healpixels (in memory when
        ``out_dir`` is None, otherwise an empty table is returned).
    """
    write_to_disk = out_dir is not None

    if pixels is None:
        pixels = np.arange(hp.nside2npix(nside))

    use_ray = initialize_use_ray(num_cpus=max_processes)

    if use_ray:
        futures = []
        test_orbits = TestOrbits.empty()

        for pixel_chunk in _iterate_chunks(pixels, chunk_size):
            futures.append(
                create_geocentric_healpixel_test_orbit_worker_remote.remote(
                    rho_bin_edges,
                    e_bin_edges,
                    nu_bin_edges,
                    psi_bin_edges,
                    pixel_chunk,
                    time,
                    origin=origin,
                    nside=nside,
                    out_dir=out_dir,
                    max_phase_angle=max_phase_angle,
                    min_q=min_q,
                    max_q=max_q,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                test_orbits_chunk = ray.get(finished[0])
                if not write_to_disk:
                    test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
                    if test_orbits.fragmented():
                        test_orbits = qv.defragment(test_orbits)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            test_orbits_chunk = ray.get(finished[0])
            if not write_to_disk:
                test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
                if test_orbits.fragmented():
                    test_orbits = qv.defragment(test_orbits)

    else:
        test_orbits = TestOrbits.empty()
        for pixel_chunk in _iterate_chunks(pixels, chunk_size):
            test_orbits_i = create_geocentric_healpixel_test_orbit_worker(
                rho_bin_edges,
                e_bin_edges,
                nu_bin_edges,
                psi_bin_edges,
                pixel_chunk,
                time,
                origin=origin,
                nside=nside,
                out_dir=out_dir,
                max_phase_angle=max_phase_angle,
                min_q=min_q,
                max_q=max_q,
            )
            if not write_to_disk:
                test_orbits = qv.concatenate([test_orbits, test_orbits_i])

    return test_orbits
