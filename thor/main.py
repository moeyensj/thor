import logging
import time
from typing import Any, Iterator, List, Optional

import quivr as qv
import ray
from adam_core.propagator import PYOORB

from .clusters import cluster_and_link
from .config import Config
from .observations.filters import ObservationFilter, TestOrbitRadiusObservationFilter
from .observations.observations import Observations
from .orbit import TestOrbit
from .orbits import (
    differential_correction,
    initial_orbit_determination,
    merge_and_extend_orbits,
)
from .range_and_transform import range_and_transform

logger = logging.getLogger("thor")


def link_test_orbit(
    test_orbit: TestOrbit,
    observations: Observations,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
) -> Iterator[Any]:
    """
    Run THOR for a single test orbit on the given observations. This function will yield
    results at each stage of the pipeline.
        1. transformed_detections
        2. clusters, cluster_members
        3. iod_orbits, iod_orbit_members
        4. od_orbits, od_orbit_members
        5. recovered_orbits, recovered_orbit_members

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbit`
        Test orbit to use to gather and transform observations.
    observations : `~thor.observations.observations.Observations`
        Observations from which range and transform the detections.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations before running THOR.
    propagator : `~adam_core.propagator.propagator.Propagator`
        Propagator to use to propagate the test orbit and generate
        ephemerides.
    max_processes : int, optional
        Maximum number of processes to use for parallelization.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    if config is None:
        config = Config()

    if config.propagator == "PYOORB":
        propagator = PYOORB
    else:
        raise ValueError(f"Unknown propagator: {config.propagator}")

    use_ray = False
    if config.max_processes is None or config.max_processes > 1:
        # Initialize ray
        if not ray.is_initialized():
            logger.debug(
                f"Ray is not initialized. Initializing with {config.max_processes}..."
            )
            ray.init(num_cpus=config.max_processes)

        if not isinstance(observations, ray.ObjectRef):
            observations = ray.put(observations)

        use_ray = True

    # Apply filters to the observations
    filtered_observations = observations
    if filters is None:
        # By default we always filter by radius from the predicted position of the test orbit
        filters = [TestOrbitRadiusObservationFilter(radius=config.cell_radius)]
    for filter_i in filters:
        filtered_observations = filter_i.apply(
            filtered_observations, test_orbit, max_processes=config.max_processes
        )

    # Defragment the observations
    if len(filtered_observations) > 0:
        filtered_observations = qv.defragment(filtered_observations)

    # Observations are no longer needed, so we can delete them
    del observations

    if use_ray:
        filtered_observations = ray.put(filtered_observations)

    # Range and transform the observations
    transformed_detections = range_and_transform(
        test_orbit,
        filtered_observations,
        propagator=propagator,
        max_processes=config.max_processes,
    )
    yield transformed_detections

    # TODO: ray support for the rest of the pipeline has not yet been implemented
    # so we will convert the ray objects to regular objects for now
    if use_ray:
        filtered_observations = ray.get(filtered_observations)

    # Run clustering
    clusters, cluster_members = cluster_and_link(
        transformed_detections,
        vx_range=[config.vx_min, config.vx_max],
        vy_range=[config.vy_min, config.vy_max],
        vx_bins=config.vx_bins,
        vy_bins=config.vy_bins,
        radius=config.cluster_radius,
        min_obs=config.cluster_min_obs,
        min_arc_length=config.cluster_min_arc_length,
        alg=config.cluster_algorithm,
        chunk_size=config.cluster_chunk_size,
        max_processes=config.max_processes,
    )
    yield clusters, cluster_members

    # Run initial orbit determination
    iod_orbits, iod_orbit_members = initial_orbit_determination(
        filtered_observations,
        cluster_members,
        min_obs=config.iod_min_obs,
        min_arc_length=config.iod_min_arc_length,
        contamination_percentage=config.iod_contamination_percentage,
        rchi2_threshold=config.iod_rchi2_threshold,
        observation_selection_method=config.iod_observation_selection_method,
        propagator=propagator,
        propagator_kwargs={},
        chunk_size=config.iod_chunk_size,
        max_processes=config.max_processes,
        # TODO: investigate whether these should be configurable
        iterate=False,
        light_time=True,
        linkage_id_col="cluster_id",
    )
    yield iod_orbits, iod_orbit_members

    # Run differential correction
    od_orbits, od_orbit_members = differential_correction(
        iod_orbits,
        iod_orbit_members,
        filtered_observations,
        min_obs=config.od_min_obs,
        min_arc_length=config.od_min_arc_length,
        contamination_percentage=config.od_contamination_percentage,
        rchi2_threshold=config.od_rchi2_threshold,
        delta=config.od_delta,
        max_iter=config.od_max_iter,
        propagator=config.propagator,
        propagator_kwargs={},
        chunk_size=config.od_chunk_size,
        max_processes=config.max_processes,
        # TODO: investigate whether these should be configurable
        method="central",
        fit_epoch=False,
    )
    yield od_orbits, od_orbit_members

    # Run arc extension
    recovered_orbits, recovered_orbit_members = merge_and_extend_orbits(
        od_orbits,
        od_orbit_members,
        filtered_observations,
        min_obs=config.arc_extension_min_obs,
        min_arc_length=config.arc_extension_min_arc_length,
        contamination_percentage=config.arc_extension_contamination_percentage,
        rchi2_threshold=config.arc_extension_rchi2_threshold,
        radius=config.arc_extension_radius,
        delta=config.od_delta,
        max_iter=config.od_max_iter,
        propagator=config.propagator,
        propagator_kwargs={},
        orbits_chunk_size=config.arc_extension_chunk_size,
        max_processes=config.max_processes,
        # TODO: investigate whether these should be configurable
        method="central",
        fit_epoch=False,
        observations_chunk_size=100000,
    )
    yield recovered_orbits, recovered_orbit_members

    time_end = time.perf_counter()
    logger.info(f"Test orbit completed in {time_end-time_start:.3f} seconds.")
