import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Optional, Tuple, Union

import quivr as qv
import ray
from adam_core.propagator.adam_pyoorb import PYOORBPropagator
from adam_core.ray_cluster import initialize_use_ray

from .checkpointing import create_checkpoint_data, load_initial_checkpoint_values
from .clusters import cluster_and_link
from .config import Config, initialize_config
from .observations.filters import ObservationFilter, filter_observations
from .observations.observations import Observations
from .orbit import TestOrbits
from .orbits import (
    differential_correction,
    initial_orbit_determination,
    merge_and_extend_orbits,
)
from .range_and_transform import range_and_transform

logger = logging.getLogger("thor")


def initialize_test_orbit(
    test_orbit: TestOrbits,
    working_dir: Optional[str] = None,
):
    """
    Initialize the test orbit by saving it to disk if a working directory is provided.
    """
    if working_dir is not None:
        test_orbit_directory = pathlib.Path(working_dir, "inputs", test_orbit.orbit_id[0].as_py())
        test_orbit_directory.mkdir(parents=True, exist_ok=True)
        test_orbit_path = os.path.join(test_orbit_directory, "test_orbit.parquet")
        test_orbit.to_parquet(test_orbit_path)


@dataclass
class LinkTestOrbitStageResult:
    """
    Result of a single stage of the THOR pipeline.
    """

    name: Literal[
        "filter_observations",
        "range_and_transform",
        "cluster_and_link",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
    ]
    result: Iterable[qv.AnyTable]
    path: Tuple[Optional[str], ...] = (None,)


def link_test_orbit(
    test_orbit: TestOrbits,
    observations: Union[str, Observations],
    working_dir: Optional[str] = None,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
) -> Iterator[LinkTestOrbitStageResult]:
    """
    Run THOR for a single test orbit on the given observations. This function will yield
    results at each stage of the pipeline.
        1. filtered observations
        2. transformed_detections
        3. clusters, cluster_members
        4. iod_orbits, iod_orbit_members
        5. od_orbits, od_orbit_members
        6. recovered_orbits, recovered_orbit_members

    Parameters
    ----------
    test_orbit : `~thor.orbit.TestOrbit`
        Test orbit to use to gather and transform observations.
    observations : `~thor.observations.observations.Observations` or str
        Observations to search for moving objects. These observations can
        be an in-memory Observations object or a path to a parquet file containing the
        observations. If a path is provided, the observations will be loaded in chunks for
        filtering.
    working_dir : str, optional
        Directory with persisted config and checkpointed results.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations before running THOR.
    config : `~thor.config.Config`, optional
        Configuration to use for THOR. If None, the default configuration will be used.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    if len(test_orbit) != 1:
        raise ValueError(f"link_test_orbit received {len(test_orbit)} orbits but expected 1.")

    test_orbit_directory = None
    if working_dir is not None:
        working_dir_path = pathlib.Path(working_dir)
        logger.info(f"Using working directory: {working_dir}")
        test_orbit_directory = pathlib.Path(working_dir, test_orbit.orbit_id[0].as_py())
        test_orbit_directory.mkdir(parents=True, exist_ok=True)
        inputs_dir = pathlib.Path(working_dir_path, "inputs")
        inputs_dir.mkdir(parents=True, exist_ok=True)

    initialize_test_orbit(test_orbit, working_dir)

    if config is None:
        config = Config()

    initialize_config(config, working_dir)

    if config.propagator == "PYOORB":
        propagator = PYOORBPropagator
    else:
        raise ValueError(f"Unknown propagator: {config.propagator}")

    use_ray = initialize_use_ray(
        num_cpus=config.max_processes,
        object_store_bytes=config.ray_memory_bytes or None,
    )

    refs_to_free = []

    checkpoint = load_initial_checkpoint_values(test_orbit_directory)
    logger.info(f"Starting at stage: {checkpoint.stage}")

    if checkpoint.stage == "complete":
        logger.info("Found recovered orbits in checkpoint, exiting early...")
        path: Tuple[Optional[str], ...] = (None,)
        if test_orbit_directory:
            path = (
                os.path.join(test_orbit_directory, "recovered_orbits.parquet"),
                os.path.join(test_orbit_directory, "recovered_orbit_members.parquet"),
            )
        return LinkTestOrbitStageResult(
            name="recover_orbits",
            result=(checkpoint.recovered_orbits, checkpoint.recovered_orbit_members),
            path=path,
        )

    if checkpoint.stage == "filter_observations":
        filtered_observations = filter_observations(observations, test_orbit, config, filters)

        filtered_observations_path = None
        if test_orbit_directory is not None:
            filtered_observations_path = os.path.join(test_orbit_directory, "filtered_observations.parquet")
            filtered_observations.to_parquet(filtered_observations_path)

        yield LinkTestOrbitStageResult(
            name="filter_observations",
            result=(filtered_observations,),
            path=(filtered_observations_path,),
        )

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")

        checkpoint = create_checkpoint_data(
            "range_and_transform",
            filtered_observations=filtered_observations,
        )

    # Observations are no longer needed
    del observations

    if checkpoint.stage == "range_and_transform":
        filtered_observations = checkpoint.filtered_observations
        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")

        # Range and transform the observations
        transformed_detections = range_and_transform(
            test_orbit,
            filtered_observations,
            propagator=propagator,
            max_processes=config.max_processes,
        )

        transformed_detections_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving transformed detections to {test_orbit_directory}...")
            transformed_detections_path = os.path.join(test_orbit_directory, "transformed_detections.parquet")
            transformed_detections.to_parquet(transformed_detections_path)

        yield LinkTestOrbitStageResult(
            name="range_and_transform",
            result=(transformed_detections,),
            path=(transformed_detections_path,),
        )

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(transformed_detections, ray.ObjectRef):
                transformed_detections = ray.put(transformed_detections)
                refs_to_free.append(transformed_detections)
                logger.info("Placed transformed detections in the object store.")

        checkpoint = create_checkpoint_data(
            "cluster_and_link",
            filtered_observations=filtered_observations,
            transformed_detections=transformed_detections,
        )

    if checkpoint.stage == "cluster_and_link":
        filtered_observations = checkpoint.filtered_observations
        transformed_detections = checkpoint.transformed_detections

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

        clusters_path = None
        cluster_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving clusters to {test_orbit_directory}...")
            clusters_path = os.path.join(test_orbit_directory, "clusters.parquet")
            cluster_members_path = os.path.join(test_orbit_directory, "cluster_members.parquet")
            clusters.to_parquet(clusters_path)
            cluster_members.to_parquet(cluster_members_path)

        yield LinkTestOrbitStageResult(
            name="cluster_and_link",
            result=(clusters, cluster_members),
            path=(clusters_path, cluster_members_path),
        )

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(clusters, ray.ObjectRef):
                clusters = ray.put(clusters)
                refs_to_free.append(clusters)
                logger.info("Placed clusters in the object store.")
            if not isinstance(cluster_members, ray.ObjectRef):
                cluster_members = ray.put(cluster_members)
                refs_to_free.append(cluster_members)
                logger.info("Placed cluster members in the object store.")

        checkpoint = create_checkpoint_data(
            "initial_orbit_determination",
            filtered_observations=filtered_observations,
            clusters=clusters,
            cluster_members=cluster_members,
        )

    if checkpoint.stage == "initial_orbit_determination":
        filtered_observations = checkpoint.filtered_observations
        clusters = checkpoint.clusters
        cluster_members = checkpoint.cluster_members

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

        iod_orbits_path = None
        iod_orbit_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving IOD orbits to {test_orbit_directory}...")
            iod_orbits_path = os.path.join(test_orbit_directory, "iod_orbits.parquet")
            iod_orbit_members_path = os.path.join(test_orbit_directory, "iod_orbit_members.parquet")
            iod_orbits.to_parquet(iod_orbits_path)
            iod_orbit_members.to_parquet(iod_orbit_members_path)

        yield LinkTestOrbitStageResult(
            name="initial_orbit_determination",
            result=(iod_orbits, iod_orbit_members),
            path=(iod_orbits_path, iod_orbit_members_path),
        )

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(iod_orbits, ray.ObjectRef):
                iod_orbits = ray.put(iod_orbits)
                refs_to_free.append(iod_orbits)
                logger.info("Placed initial orbits in the object store.")
            if not isinstance(iod_orbit_members, ray.ObjectRef):
                iod_orbit_members = ray.put(iod_orbit_members)
                refs_to_free.append(iod_orbit_members)
                logger.info("Placed initial orbit members in the object store.")

        checkpoint = create_checkpoint_data(
            "differential_correction",
            filtered_observations=filtered_observations,
            iod_orbits=iod_orbits,
            iod_orbit_members=iod_orbit_members,
        )

    if checkpoint.stage == "differential_correction":
        filtered_observations = checkpoint.filtered_observations
        iod_orbits = checkpoint.iod_orbits
        iod_orbit_members = checkpoint.iod_orbit_members

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
            propagator=propagator,
            propagator_kwargs={},
            chunk_size=config.od_chunk_size,
            max_processes=config.max_processes,
            # TODO: investigate whether these should be configurable
            method="central",
        )

        od_orbits_path = None
        od_orbit_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving OD orbits to {test_orbit_directory}...")
            od_orbits_path = os.path.join(test_orbit_directory, "od_orbits.parquet")
            od_orbit_members_path = os.path.join(test_orbit_directory, "od_orbit_members.parquet")
            od_orbits.to_parquet(od_orbits_path)
            od_orbit_members.to_parquet(od_orbit_members_path)

        yield LinkTestOrbitStageResult(
            name="differential_correction",
            result=(od_orbits, od_orbit_members),
            path=(od_orbits_path, od_orbit_members_path),
        )

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                refs_to_free.append(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(od_orbits, ray.ObjectRef):
                od_orbits = ray.put(od_orbits)
                refs_to_free.append(od_orbits)
                logger.info("Placed differentially corrected orbits in the object store.")
            if not isinstance(od_orbit_members, ray.ObjectRef):
                od_orbit_members = ray.put(od_orbit_members)
                refs_to_free.append(od_orbit_members)
                logger.info("Placed differentially corrected orbit members in the object store.")

        checkpoint = create_checkpoint_data(
            "recover_orbits",
            filtered_observations=filtered_observations,
            od_orbits=od_orbits,
            od_orbit_members=od_orbit_members,
        )

    if checkpoint.stage == "recover_orbits":
        filtered_observations = checkpoint.filtered_observations
        od_orbits = checkpoint.od_orbits
        od_orbit_members = checkpoint.od_orbit_members

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
            propagator=propagator,
            propagator_kwargs={},
            orbits_chunk_size=config.arc_extension_chunk_size,
            max_processes=config.max_processes,
            # TODO: investigate whether these should be configurable
            method="central",
            observations_chunk_size=100000,
        )

        if use_ray and len(refs_to_free) > 0:
            ray.internal.free(refs_to_free)
            logger.info(f"Removed {len(refs_to_free)} references from the object store.")

        recovered_orbits_path = None
        recovered_orbit_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving recovered orbits to {test_orbit_directory}...")
            recovered_orbits_path = os.path.join(test_orbit_directory, "recovered_orbits.parquet")
            recovered_orbit_members_path = os.path.join(
                test_orbit_directory, "recovered_orbit_members.parquet"
            )
            recovered_orbits.to_parquet(recovered_orbits_path)
            recovered_orbit_members.to_parquet(recovered_orbit_members_path)

        time_end = time.perf_counter()
        logger.info(f"Test orbit completed in {time_end-time_start:.3f} seconds.")
        yield LinkTestOrbitStageResult(
            name="recover_orbits",
            result=(recovered_orbits, recovered_orbit_members),
            path=(recovered_orbits_path, recovered_orbit_members_path),
        )
