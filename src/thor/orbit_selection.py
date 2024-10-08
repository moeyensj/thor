import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.coordinates import KeplerianCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import Propagator
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.propagator.utils import _iterate_chunks
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from thor.observations import Observations
from thor.orbit import TestOrbits

from .observations.utils import calculate_healpixels

logger = logging.getLogger(__name__)

__all__ = ["generate_test_orbits"]


@dataclass
class KeplerianPhaseSpace:
    a_min: float = -1_000_000.0
    a_max: float = 1_000_000.0
    e_min: float = 0.0
    e_max: float = 1_000.0
    i_min: float = 0.0
    i_max: float = 180.0


def select_average_within_region(coordinates: KeplerianCoordinates) -> int:
    """
    Select the Keplerian coordinate as close to the median in semi-major axis,
    eccentricity, and inclination.

    Parameters
    ----------
    coordinates
        Keplerian coordinates to select from.

    Returns
    -------
    index
        Index of the selected coordinates.
    """
    keplerian = coordinates.values
    aei = keplerian[:, 0:3]

    median = np.median(aei, axis=0)
    percent_diff = np.abs((aei - median) / median)

    # Sum the percent differences
    summed_diff = np.sum(percent_diff, axis=1)

    # Find the minimum summed percent difference and call that
    # the average object
    index = np.where(summed_diff == np.min(summed_diff))[0][0]
    return index


def select_test_orbits(ephemeris: Ephemeris, orbits: Orbits) -> Orbits:
    """
    Select test orbits from orbits using the predicted ephemeris
    for different regions of Keplerian phase space.

    The regions are:
    - 3 in the Hungarias
    - 5 in the main belt
    - 1 in the outer solar system

    Parameters
    ----------
    ephemeris
        Ephemeris for the orbits.
    orbits
        Orbits to select from.

    Returns
    -------
    test_orbits
        Test orbits selected from the orbits.
    """
    orbits_patch = orbits.apply_mask(pc.is_in(orbits.orbit_id, ephemeris.orbit_id))

    # Convert to keplerian coordinates
    keplerian = orbits_patch.coordinates.to_keplerian()

    # Create 3 phase space regions for the Hungarias
    hungarias_01 = KeplerianPhaseSpace(
        a_min=1.7,
        a_max=2.06,
        e_max=0.1,
    )
    hungarias_02 = KeplerianPhaseSpace(
        a_min=hungarias_01.a_min,
        a_max=hungarias_01.a_max,
        e_min=hungarias_01.e_max,
        e_max=0.2,
    )
    hungarias_03 = KeplerianPhaseSpace(
        a_min=hungarias_01.a_min,
        a_max=hungarias_01.a_max,
        e_min=hungarias_02.e_max,
        e_max=0.4,
    )

    # Create 5 phase space regions for the rest of the main belt
    mainbelt_01 = KeplerianPhaseSpace(
        a_min=hungarias_03.a_max,
        a_max=2.5,
        e_max=0.5,
    )
    mainbelt_02 = KeplerianPhaseSpace(
        a_min=mainbelt_01.a_max,
        a_max=2.82,
        e_max=0.5,
    )
    mainbelt_03 = KeplerianPhaseSpace(
        a_min=mainbelt_02.a_max,
        a_max=2.95,
        e_max=0.5,
    )
    mainbelt_04 = KeplerianPhaseSpace(
        a_min=mainbelt_03.a_max,
        a_max=3.27,
        e_max=0.5,
    )
    mainbelt_05 = KeplerianPhaseSpace(
        a_min=mainbelt_04.a_max,
        a_max=5.0,
        e_max=0.5,
    )

    # Create 1 phase space region for trojans, TNOs, etc..
    outer = KeplerianPhaseSpace(
        a_min=mainbelt_05.a_max,
        a_max=50.0,
        e_max=0.5,
    )

    phase_space_regions = [
        hungarias_01,
        hungarias_02,
        hungarias_03,
        mainbelt_01,
        mainbelt_02,
        mainbelt_03,
        mainbelt_04,
        mainbelt_05,
        outer,
    ]

    test_orbits = []
    for region in phase_space_regions:
        mask = pc.and_(
            pc.and_(
                pc.and_(
                    pc.and_(
                        pc.and_(
                            pc.greater_equal(keplerian.a, region.a_min),
                            pc.less(keplerian.a, region.a_max),
                        ),
                        pc.greater_equal(keplerian.e, region.e_min),
                    ),
                    pc.less(keplerian.e, region.e_max),
                ),
                pc.greater_equal(keplerian.i, region.i_min),
            ),
            pc.less(keplerian.i, region.i_max),
        )

        keplerian_region = keplerian.apply_mask(mask)
        orbits_region = orbits_patch.apply_mask(mask)

        if len(keplerian_region) != 0:
            index = select_average_within_region(keplerian_region)
            test_orbits.append(orbits_region[int(index)])

    if len(test_orbits) > 0:
        return qv.concatenate(test_orbits)
    else:
        return Orbits.empty()


def generate_test_orbits_worker(
    healpixel_chunk: pa.Array,
    ephemeris_healpixels: pa.Array,
    propagated_orbits: Union[Orbits, ray.ObjectRef],
    ephemeris: Union[Ephemeris, ray.ObjectRef],
) -> TestOrbits:
    """
    Worker function for generating test orbits.

    Parameters
    ----------
    healpixel_chunk
        Healpixels to generate test orbits for.
    ephemeris_healpixels
        Healpixels for the ephemeris.
    propagated_orbits
        Propagated orbits.
    ephemeris
        Ephemeris for the propagated orbits.

    Returns
    -------
    test_orbits
        Test orbits generated from the propagated orbits.
    """
    test_orbits_list = []

    # Filter the ephemerides to only those in the observations
    ephemeris_mask = pc.is_in(ephemeris_healpixels, healpixel_chunk)
    ephemeris_filtered = ephemeris.apply_mask(ephemeris_mask)
    ephemeris_healpixels = pc.filter(ephemeris_healpixels, ephemeris_mask)
    logger.info(f"{len(ephemeris_filtered)} orbit ephemerides overlap with the observations.")

    # Filter the orbits to only those in the ephemeris
    orbits_filtered = propagated_orbits.apply_mask(
        pc.is_in(propagated_orbits.orbit_id, ephemeris_filtered.orbit_id)
    )

    logger.info("Selecting test orbits from the orbit catalog...")
    for healpixel in healpixel_chunk:
        healpixel_mask = pc.equal(ephemeris_healpixels, healpixel)
        ephemeris_healpixel = ephemeris_filtered.apply_mask(healpixel_mask)

        if len(ephemeris_healpixel) == 0:
            logger.debug(f"No ephemerides in healpixel {healpixel}.")
            continue

        test_orbits_healpixel = select_test_orbits(ephemeris_healpixel, orbits_filtered)

        if len(test_orbits_healpixel) > 0:
            test_orbits_list.append(
                TestOrbits.from_kwargs(
                    orbit_id=test_orbits_healpixel.orbit_id,
                    object_id=test_orbits_healpixel.object_id,
                    coordinates=test_orbits_healpixel.coordinates,
                    bundle_id=[healpixel for _ in range(len(test_orbits_healpixel))],
                )
            )
        else:
            logger.debug(f"No orbits in healpixel {healpixel}.")

    if len(test_orbits_list) > 0:
        test_orbits = qv.concatenate(test_orbits_list)
    else:
        test_orbits = TestOrbits.empty()

    return test_orbits


generate_test_orbits_worker_remote = ray.remote(generate_test_orbits_worker)
generate_test_orbits_worker_remote.options(num_cpus=1, num_returns=1)


def generate_test_orbits(
    observations: Union[str, Observations],
    catalog: Orbits,
    nside: int = 32,
    propagator: Propagator = PYOORBPropagator(),
    max_processes: Optional[int] = None,
    chunk_size: int = 100,
) -> TestOrbits:
    """
    Given observations and a catalog of known orbits generate test orbits
    from the catalog. The observations are divded into healpixels (with size determined
    by the nside parameter). For each healpixel in observations, select up to 9 orbits from
    the catalog that are in the same healpixel as the observations. The orbits are selected
    in bins of semi-major axis, eccentricity, and inclination.

    The catalog will be propagated to start time of the observations using the propagator
    and ephemerides will be generated for the propagated orbits (assuming a geocentric observer).

    Parameters
    ----------
    observations
        Observations for which to generate test orbits. These observations can
        be an in-memory Observations object or a path to a parquet file containing the
        observations.
    catalog
        Catalog of known orbits.
    nside
        Healpixel size.
    propagator
        Propagator to use to propagate the orbits.
    max_processes
        Maximum number of processes to use while propagating orbits and
        generating ephemerides.
    chunk_size
        The maximum number of unique healpixels for which to generate test orbits per
        process. This function will dynamically compute the chunk size based on the
        number of unique healpixels and the number of processes. The dynamic chunk
        size will never exceed the given value.

    Returns
    -------
    test_orbits
        Test orbits generated from the catalog.
    """
    time_start = time.perf_counter()
    logger.info("Generating test orbits...")

    # If the input file is a string, read in the days column to
    # extract the minimum time
    if isinstance(observations, str):
        table = pq.read_table(observations, columns=["coordinates.time.days"], memory_map=True)

        min_day = pc.min(table["days"]).as_py()
        # Set the start time to the midnight of the first night of observations
        start_time = Timestamp.from_kwargs(days=[min_day], nanos=[0], scale="utc")
        del table
    elif isinstance(observations, Observations):
        # Extract the minimum time from the observations
        earliest_time = observations.coordinates.time.min()

        # Set the start time to the midnight of the first night of observations
        start_time = Timestamp.from_kwargs(days=earliest_time.days, nanos=[0], scale="utc")
    else:
        raise ValueError(
            f"observations must be a path to a parquet file or an Observations object. Got {type(observations)}."
        )

    # Propagate the orbits to the minimum time
    logger.info("Propagating orbits to the start time of the observations...")
    propagation_start_time = time.perf_counter()
    propagated_orbits = propagator.propagate_orbits(
        catalog,
        start_time,
        max_processes=max_processes,
        chunk_size=500,
    )
    propagation_end_time = time.perf_counter()
    logger.info(f"Propagation completed in {propagation_end_time - propagation_start_time:.3f} seconds.")

    # Create a geocentric observer for the observations
    logger.info("Generating ephemerides for the propagated orbits...")
    ephemeris_start_time = time.perf_counter()
    observers = Observers.from_code("500", start_time)

    # Generate ephemerides for the propagated orbits
    ephemeris = propagator.generate_ephemeris(
        propagated_orbits,
        observers,
        start_time,
        max_processes=max_processes,
        chunk_size=1000,
    )
    ephemeris_end_time = time.perf_counter()
    logger.info(f"Ephemeris generation completed in {ephemeris_end_time - ephemeris_start_time:.3f} seconds.")

    if isinstance(observations, str):
        table = pq.read_table(
            observations,
            columns=["coordinates.lon", "coordinates.lat"],
            memory_map=True,
        )
        lon = table["lon"].to_numpy(zero_copy_only=False)
        lat = table["lat"].to_numpy(zero_copy_only=False)
        del table

    else:
        lon = observations.coordinates.lon.to_numpy(zero_copy_only=False)
        lat = observations.coordinates.lat.to_numpy(zero_copy_only=False)

    # Calculate the healpixels for observations and ephemerides
    # Here we want the unique healpixels so we can cross match against our
    # catalog's predicted ephemeris
    observations_healpixels = calculate_healpixels(
        lon,
        lat,
        nside=nside,
    )
    observations_healpixels = pc.unique(pa.array(observations_healpixels))
    logger.info(f"Observations occur in {len(observations_healpixels)} unique healpixels.")

    # Calculate the healpixels for each ephemeris
    # We do not want unique healpixels here because we want to
    # select orbits from the same healpixel as the observations
    ephemeris_healpixels = calculate_healpixels(
        ephemeris.coordinates.lon.to_numpy(zero_copy_only=False),
        ephemeris.coordinates.lat.to_numpy(zero_copy_only=False),
        nside=nside,
    )
    ephemeris_healpixels = pa.array(ephemeris_healpixels)

    # Dynamically compute the chunk size based on the number of healpixels
    # and the number of processes
    if max_processes is None:
        max_processes = mp.cpu_count()

    chunk_size = np.minimum(np.ceil(len(observations_healpixels) / max_processes).astype(int), chunk_size)
    logger.info(f"Generating test orbits with a chunk size of {chunk_size} healpixels.")

    test_orbits = TestOrbits.empty()
    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:

        ephemeris_ref = ray.put(ephemeris)
        ephemeris_healpixels_ref = ray.put(ephemeris_healpixels)
        propagated_orbits_ref = ray.put(propagated_orbits)

        futures = []
        for healpixel_chunk in _iterate_chunks(observations_healpixels, chunk_size):
            futures.append(
                generate_test_orbits_worker_remote.remote(
                    healpixel_chunk,
                    ephemeris_healpixels_ref,
                    propagated_orbits_ref,
                    ephemeris_ref,
                )
            )

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            test_orbits = qv.concatenate([test_orbits, ray.get(finished[0])])
            if test_orbits.fragmented():
                test_orbits = qv.defragment(test_orbits)

    else:

        for healpixel_chunk in _iterate_chunks(observations_healpixels, chunk_size):
            test_orbits_chunk = generate_test_orbits_worker(
                healpixel_chunk,
                ephemeris_healpixels,
                propagated_orbits,
                ephemeris,
            )
            test_orbits = qv.concatenate([test_orbits, test_orbits_chunk])
            if test_orbits.fragmented():
                test_orbits = qv.defragment(test_orbits)

    time_end = time.perf_counter()
    logger.info(f"Selected {len(test_orbits)} test orbits.")
    logger.info(f"Test orbit generation completed in {time_end - time_start:.3f} seconds.")
    return test_orbits
