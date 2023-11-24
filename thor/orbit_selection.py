import logging
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import KeplerianCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import PYOORB, Propagator

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


def generate_test_orbits(
    observations: Observations,
    catalog: Orbits,
    nside: int = 32,
    propagator: Propagator = PYOORB(),
    max_processes: int = 1,
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
        Observations to generate test orbits for.
    catalog
        Catalog of known orbits.
    nside
        Healpixel size.
    propagator
        Propagator to use to propagate the orbits.
    max_processes
        Maximum number of processes to use while propagating orbits and
        generating ephemerides.

    Returns
    -------
    test_orbits
        Test orbits generated from the catalog.
    """
    # Extract the minimum time from the observations
    start_time = observations.coordinates.time.min()

    # Propagate the orbits to the minimum time
    logger.info("Propagating orbits to the start time of the observations...")
    propagation_start_time = time.perf_counter()
    propagated_orbits = propagator.propagate_orbits(
        catalog,
        start_time,
        max_processes=max_processes,
        parallel_backend="ray",
        chunk_size=500,
    )
    propagation_end_time = time.perf_counter()
    logger.info(
        f"Propagation completed in {propagation_end_time - propagation_start_time:.3f} seconds."
    )

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
        parallel_backend="ray",
        chunk_size=1000,
    )
    ephemeris_end_time = time.perf_counter()
    logger.info(
        f"Ephemeris generation completed in {ephemeris_end_time - ephemeris_start_time:.3f} seconds."
    )

    # Calculate the healpixels for observations and ephemerides
    observations_healpixels = calculate_healpixels(
        observations.coordinates.lon.to_numpy(zero_copy_only=False),
        observations.coordinates.lat.to_numpy(zero_copy_only=False),
        nside=nside,
    )
    observations_healpixels = np.unique(observations_healpixels)
    logger.info(
        f"Observations occur in {len(observations_healpixels)} unique healpixels."
    )

    # Calculate the healpixels for the ephemerides
    ephemeris_healpixels = calculate_healpixels(
        ephemeris.coordinates.lon.to_numpy(zero_copy_only=False),
        ephemeris.coordinates.lat.to_numpy(zero_copy_only=False),
        nside=nside,
    )

    # Filter the ephemerides to only those in the observations
    ephemeris_mask = pa.array(np.in1d(ephemeris_healpixels, observations_healpixels))
    ephemeris_filtered = ephemeris.apply_mask(ephemeris_mask)
    ephemeris_healpixels = ephemeris_healpixels[
        ephemeris_mask.to_numpy(zero_copy_only=False)
    ]
    logger.info(
        f"{len(ephemeris_filtered)} orbit ephemerides overlap with the observations."
    )

    # Filter the orbits to only those in the ephemeris
    orbits_filtered = propagated_orbits.apply_mask(
        pc.is_in(propagated_orbits.orbit_id, ephemeris_filtered.orbit_id)
    )

    logger.info("Selecting test orbits from the orbit catalog...")
    test_orbits_list = []
    for healpixel in observations_healpixels:
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

    logger.info(f"Selected {len(test_orbits)} test orbits.")
    return test_orbits
