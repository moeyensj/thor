import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import transform_coordinates
from adam_core.coordinates.residuals import Residuals
from adam_core.orbits import Orbits
from adam_core.ray_cluster import initialize_use_ray
from adam_core.utils.iter import _iterate_chunks

from ..orbit import TestOrbits


class NearestTestOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn()
    test_orbit_id = qv.LargeStringColumn(nullable=True)
    bundle_id = qv.Int64Column(nullable=True)
    residuals = Residuals.as_column(nullable=True)


class OrbitHealpixel(qv.Table):
    orbit_id = qv.LargeStringColumn()
    healpixel = qv.Int64Column()
    nside = qv.IntAttribute()

    @classmethod
    def from_orbits(cls, orbits: Orbits, nside: int = 64) -> "OrbitHealpixel":
        """
        Calculate the healpixels for a set of orbits.

        Parameters
        ----------
        orbits : Orbits (N)
            The orbits to calculate the healpixels for.
        nside : int, optional
            The nside of the healpixel grid. Default is 64.

        Returns
        -------
        OrbitHealpixel (N)
            The healpixels for the orbits.
        """
        r_hat = orbits.coordinates.r_hat
        x = r_hat[:, 0]
        y = r_hat[:, 1]
        z = r_hat[:, 2]

        healpixels = hp.vec2pix(nside, x, y, z, nest=True)
        return OrbitHealpixel.from_kwargs(
            orbit_id=orbits.orbit_id,
            healpixel=healpixels,
            nside=nside,
        )


def nearest_test_orbit_worker(
    pixels: np.ndarray,
    orbit_healpixels: OrbitHealpixel,
    test_orbits_healpixels: OrbitHealpixel,
    orbits: Orbits,
    test_orbits: TestOrbits,
) -> NearestTestOrbits:
    """
    Worker function for finding the nearest test orbit for each orbit.

    Parameters
    ----------
    pixels : np.ndarray
        The pixels to find the nearest test orbit for.
    orbit_healpixels : OrbitHealpixel
        The healpixels for the orbits.
    test_orbits_healpixels : OrbitHealpixel
        The healpixels for the test orbits.
    orbits : Orbits
        The orbits to find the nearest test orbit for.
    test_orbits : TestOrbits
        The test orbits to find the nearest test orbit for.

    Returns
    -------
    NearestTestOrbits
        The nearest test orbit for each orbit.
    """
    nearest_test_orbits = NearestTestOrbits.empty()
    for pixel in pixels:

        orbit_ids_healpixel = orbit_healpixels.select("healpixel", pixel).orbit_id
        test_orbit_ids_healpixel = test_orbits_healpixels.select("healpixel", pixel).orbit_id
        test_orbits_healpixel_filtered = test_orbits.apply_mask(
            pc.is_in(test_orbits.orbit_id, test_orbit_ids_healpixel)
        )

        if len(test_orbits_healpixel_filtered) == 0:

            nearest_test_orbit_i = NearestTestOrbits.from_kwargs(
                orbit_id=orbit_ids_healpixel,
                test_orbit_id=None,
                bundle_id=None,
                residuals=None,
            )
            nearest_test_orbits = qv.concatenate([nearest_test_orbits, nearest_test_orbit_i])

        else:

            for orbit_id in orbit_ids_healpixel:

                residuals = Residuals.calculate(
                    test_orbits_healpixel_filtered.coordinates,
                    orbits.select("orbit_id", orbit_id).coordinates,
                )

                ind = np.argmin(residuals.chi2.to_numpy(zero_copy_only=False))
                test_orbit_nearest = test_orbits_healpixel_filtered.take([ind])

                nearest_test_orbit_i = NearestTestOrbits.from_kwargs(
                    orbit_id=pa.repeat(orbit_id, len(test_orbit_nearest)),
                    test_orbit_id=test_orbit_nearest.orbit_id,
                    bundle_id=test_orbit_nearest.bundle_id,
                    residuals=residuals.take([ind]),
                )

                nearest_test_orbits = qv.concatenate([nearest_test_orbits, nearest_test_orbit_i])

    return nearest_test_orbits


nearest_test_orbit_worker_ray = ray.remote(nearest_test_orbit_worker)


def find_nearest_test_orbit(
    orbits: Orbits, test_orbits: TestOrbits, nside: int = 64, chunk_size: int = 10, max_processes: int = 30
) -> NearestTestOrbits:
    """
    Find the nearest test orbit for each orbit. The orbits and test orbits are initially filtered
    to only include those in the same healpixels as the orbits. Then the nearest test orbit is found
    for each orbit by calculating the residuals between the orbit and the test orbits.

    Parameters
    ----------
    orbits : Orbits
        The orbits to find the nearest test orbit for.
    test_orbits : TestOrbits
        The test orbits to find the nearest test orbit for.
    nside : int, optional
        The nside of the healpixel grid. Default is 64.
    chunk_size : int, optional
        The chunk size for the healpixel grid. Default is 10.
    max_processes : int, optional
        The maximum number of processes to use. Default is 30.

    Returns
    -------
    NearestTestOrbits
        The nearest test orbit for each orbit.
    """
    if test_orbits.fragmented():
        test_orbits = qv.defragment(test_orbits)
    if orbits.fragmented():
        orbits = qv.defragment(orbits)

    # Transform orbits and test orbits to ecliptic barycentric coordinates
    test_orbits = test_orbits.set_column(
        "coordinates", transform_coordinates(test_orbits.coordinates, frame_out="ecliptic")
    )
    orbits = orbits.set_column("coordinates", transform_coordinates(orbits.coordinates, frame_out="ecliptic"))

    # Calculate the healpixels for the orbits and test orbits (uses the
    # unit barycentric position unit vector)
    test_orbits_healpixels = OrbitHealpixel.from_orbits(test_orbits, nside)
    orbits_healpixels = OrbitHealpixel.from_orbits(orbits, nside)

    # Filter test orbits to only include those in the same healpixels as the orbits
    unique_orbit_healpixels = orbits_healpixels.healpixel.unique()
    test_orbits_healpixels_filtered = test_orbits_healpixels.apply_mask(
        pc.is_in(test_orbits_healpixels.healpixel, unique_orbit_healpixels)
    )
    test_orbits_filtered = test_orbits.apply_mask(
        pc.is_in(test_orbits.orbit_id, test_orbits_healpixels_filtered.orbit_id)
    )

    use_ray = initialize_use_ray(num_cpus=max_processes)
    nearest_test_orbits = NearestTestOrbits.empty()

    if use_ray:
        futures = []

        test_orbits_ref = ray.put(test_orbits_filtered)
        orbits_ref = ray.put(orbits)
        orbit_healpixels_ref = ray.put(orbits_healpixels)
        test_orbits_healpixels_filtered_ref = ray.put(test_orbits_healpixels_filtered)

        for pixels in _iterate_chunks(unique_orbit_healpixels, chunk_size):
            futures.append(
                nearest_test_orbit_worker_ray.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=True,
                        _spill_on_unavailable=True,
                    ),
                ).remote(
                    pixels,
                    orbit_healpixels_ref,
                    test_orbits_healpixels_filtered_ref,
                    orbits_ref,
                    test_orbits_ref,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                nearest_test_orbits_chunk = ray.get(finished[0])
                nearest_test_orbits = qv.concatenate([nearest_test_orbits, nearest_test_orbits_chunk])
                if nearest_test_orbits.fragmented():
                    nearest_test_orbits = qv.defragment(nearest_test_orbits)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            nearest_test_orbits_chunk = ray.get(finished[0])
            nearest_test_orbits = qv.concatenate([nearest_test_orbits, nearest_test_orbits_chunk])
            if nearest_test_orbits.fragmented():
                nearest_test_orbits = qv.defragment(nearest_test_orbits)

    else:

        for pixels in _iterate_chunks(unique_orbit_healpixels, chunk_size):

            nearest_test_orbits_chunk = nearest_test_orbit_worker(
                pixels, orbits_healpixels, test_orbits_healpixels_filtered, orbits, test_orbits_filtered
            )
            nearest_test_orbits = qv.concatenate([nearest_test_orbits, nearest_test_orbits_chunk])

    return nearest_test_orbits
