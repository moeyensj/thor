import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Literal, Optional, Union

import quivr as qv
import ray
from adam_core.coordinates import CartesianCoordinates
from adam_core.propagator import PYOORB
from adam_core.time import Timestamp

from .clusters import ClusterMembers, Clusters, cluster_and_link
from .config import Config, initialize_config
from .observations.filters import ObservationFilter, filter_observations
from .observations.observations import Observations
from .orbit import TestOrbit
from .orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from .orbits import (
    differential_correction,
    initial_orbit_determination,
    merge_and_extend_orbits,
)
from .range_and_transform import TransformedDetections, range_and_transform

logger = logging.getLogger("thor")


def initialize_use_ray(config: Config) -> bool:
    use_ray = False
    if config.max_processes is None or config.max_processes > 1:
        # Initialize ray
        if not ray.is_initialized():
            logger.debug(
                f"Ray is not initialized. Initializing with {config.max_processes}..."
            )
            ray.init(num_cpus=config.max_processes)

        use_ray = True
    return use_ray


@dataclass
class CheckpointData:
    stage: Literal[
        "filter_observations",
        "range_and_transform",
        "cluster_and_link",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
        "complete",
    ]
    filtered_observations: Optional[Observations] = None
    transformed_detections: Optional[TransformedDetections] = None
    clusters: Optional[Clusters] = None
    cluster_members: Optional[ClusterMembers] = None
    iod_orbits: Optional[FittedOrbits] = None
    iod_orbit_members: Optional[FittedOrbitMembers] = None
    od_orbits: Optional[FittedOrbits] = None
    od_orbit_members: Optional[FittedOrbitMembers] = None
    recovered_orbits: Optional[FittedOrbits] = None
    recovered_orbit_members: Optional[FittedOrbitMembers] = None


def initialize_test_orbit(
    test_orbit: TestOrbit,
    working_dir: Optional[str] = None,
):
    """
    Initialize the test orbit by saving it to disk if a working directory is provided.
    """
    if working_dir is not None:
        test_orbit_directory = pathlib.Path(working_dir, "inputs", test_orbit.orbit_id)
        test_orbit_directory.mkdir(parents=True, exist_ok=True)
        test_orbit_path = os.path.join(test_orbit_directory, "test_orbit.parquet")
        test_orbit.orbit.to_parquet(test_orbit_path)


def load_initial_checkpoint_values(
    test_orbit_directory: Optional[pathlib.Path] = None,
) -> CheckpointData:
    """
    Check for completed stages and return values from disk if they exist.

    We want to avoid loading objects into memory that are not required.
    """
    stage = "filter_observations"
    # Without a checkpoint directory, we always start at the beginning
    if test_orbit_directory is None:
        return CheckpointData(stage=stage)

    # filtered_observations is always needed when it exists
    filtered_observations_path = pathlib.Path(
        test_orbit_directory, "filtered_observations.parquet"
    )
    # If it doesn't exist, start at the beginning.
    if not filtered_observations_path.exists():
        return CheckpointData(stage=stage)
    logger.info("Found filtered observations")
    filtered_observations = Observations.from_parquet(filtered_observations_path)

    # Unfortunately we have to reinitialize the times to set the attribute
    # correctly.
    filtered_observations = qv.defragment(filtered_observations)
    filtered_observations = filtered_observations.sort_by(
        [
            "detections.time.days",
            "detections.time.nanos",
            "observatory_code",
        ]
    )

    # If the pipeline was started but we have recovered_orbits already, we
    # are done and should exit early.
    recovered_orbits_path = pathlib.Path(
        test_orbit_directory, "recovered_orbits.parquet"
    )
    recovered_orbit_members_path = pathlib.Path(
        test_orbit_directory, "recovered_orbit_members.parquet"
    )
    if recovered_orbits_path.exists() and recovered_orbit_members_path.exists():
        logger.info("Found recovered orbits in checkpoint")
        recovered_orbits = FittedOrbits.from_parquet(recovered_orbits_path)
        recovered_orbit_members = FittedOrbitMembers.from_parquet(
            recovered_orbit_members_path
        )

        # Unfortunately we have to reinitialize the times to set the attribute
        # correctly.
        recovered_orbits = qv.defragment(recovered_orbits)
        recovered_orbits = recovered_orbits.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
            ]
        )

        return CheckpointData(
            stage="complete",
            recovered_orbits=recovered_orbits,
            recovered_orbit_members=recovered_orbit_members,
        )

    # Now with filtered_observations available, we can check for the later
    # stages in reverse order.
    od_orbits_path = pathlib.Path(test_orbit_directory, "od_orbits.parquet")
    od_orbit_members_path = pathlib.Path(
        test_orbit_directory, "od_orbit_members.parquet"
    )
    if od_orbits_path.exists() and od_orbit_members_path.exists():
        logger.info("Found OD orbits in checkpoint")
        od_orbits = FittedOrbits.from_parquet(od_orbits_path)
        od_orbit_members = FittedOrbitMembers.from_parquet(od_orbit_members_path)

        # Unfortunately we have to reinitialize the times to set the attribute
        # correctly.
        od_orbits = qv.defragment(od_orbits)
        od_orbits = od_orbits.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
            ]
        )

        return CheckpointData(
            stage="recover_orbits",
            filtered_observations=filtered_observations,
            od_orbits=od_orbits,
            od_orbit_members=od_orbit_members,
        )

    iod_orbits_path = pathlib.Path(test_orbit_directory, "iod_orbits.parquet")
    iod_orbit_members_path = pathlib.Path(
        test_orbit_directory, "iod_orbit_members.parquet"
    )
    if iod_orbits_path.exists() and iod_orbit_members_path.exists():
        logger.info("Found IOD orbits")
        iod_orbits = FittedOrbits.from_parquet(iod_orbits_path)
        iod_orbit_members = FittedOrbitMembers.from_parquet(iod_orbit_members_path)

        # Unfortunately we have to reinitialize the times to set the attribute
        # correctly.
        iod_orbits = qv.defragment(iod_orbits)
        iod_orbits = iod_orbits.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
            ]
        )

        return CheckpointData(
            stage="differential_correction",
            filtered_observations=filtered_observations,
            iod_orbits=iod_orbits,
            iod_orbit_members=iod_orbit_members,
        )

    clusters_path = pathlib.Path(test_orbit_directory, "clusters.parquet")
    cluster_members_path = pathlib.Path(test_orbit_directory, "cluster_members.parquet")
    if clusters_path.exists() and cluster_members_path.exists():
        logger.info("Found clusters")
        clusters = Clusters.from_parquet(clusters_path)
        cluster_members = ClusterMembers.from_parquet(cluster_members_path)

        return CheckpointData(
            stage="initial_orbit_determination",
            filtered_observations=filtered_observations,
            clusters=clusters,
            cluster_members=cluster_members,
        )

    transformed_detections_path = pathlib.Path(
        test_orbit_directory, "transformed_detections.parquet"
    )
    if transformed_detections_path.exists():
        logger.info("Found transformed detections")
        transformed_detections = TransformedDetections.from_parquet(
            transformed_detections_path
        )

        return CheckpointData(
            stage="cluster_and_link",
            filtered_observations=filtered_observations,
            transformed_detections=transformed_detections,
        )

    return CheckpointData(
        stage="range_and_transform", filtered_observations=filtered_observations
    )


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
    result: Iterable[Any]
    path: Optional[Iterable[str]] = None


def link_test_orbit(
    test_orbit: TestOrbit,
    observations: Observations,
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
    observations : `~thor.observations.observations.Observations`
        Observations from which range and transform the detections.
    working_dir : str, optional
        Directory with persisted config and checkpointed results.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations before running THOR.
    config : `~thor.config.Config`, optional
        Configuration to use for THOR. If None, the default configuration will be used.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    test_orbit_directory = None
    if working_dir is not None:
        working_dir = pathlib.Path(working_dir)
        logger.info(f"Using working directory: {working_dir}")
        test_orbit_directory = pathlib.Path(working_dir, test_orbit.orbit_id)
        test_orbit_directory.mkdir(parents=True, exist_ok=True)
        inputs_dir = pathlib.Path(working_dir, "inputs")
        inputs_dir.mkdir(parents=True, exist_ok=True)

    initialize_test_orbit(test_orbit, working_dir)

    if config is None:
        config = Config()

    initialize_config(config, working_dir)

    if config.propagator == "PYOORB":
        propagator = PYOORB
    else:
        raise ValueError(f"Unknown propagator: {config.propagator}")

    use_ray = initialize_use_ray(config)

    if (
        use_ray
        and observations is not None
        and not isinstance(observations, ray.ObjectRef)
    ):
        observations = ray.put(observations)

    checkpoint = load_initial_checkpoint_values(test_orbit_directory)
    logger.info(f"Starting at stage: {checkpoint.stage}")

    if checkpoint.stage == "complete":
        logger.info("Found recovered orbits in checkpoint, exiting early...")
        path = None
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
        filtered_observations = filter_observations(
            observations, test_orbit, config, filters
        )

        filtered_observations_path = None
        if test_orbit_directory is not None:
            filtered_observations_path = os.path.join(
                test_orbit_directory, "filtered_observations.parquet"
            )
            filtered_observations.to_parquet(filtered_observations_path)

        yield LinkTestOrbitStageResult(
            name="filter_observations",
            result=(filtered_observations,),
            path=(filtered_observations_path,),
        )

        checkpoint = CheckpointData(
            stage="range_and_transform",
            filtered_observations=filtered_observations,
        )

    # Observations are no longer needed, so we can delete them
    del observations

    if checkpoint.stage == "range_and_transform":
        filtered_observations = checkpoint.filtered_observations
        if use_ray and not isinstance(filtered_observations, ray.ObjectRef):
            filtered_observations = ray.put(filtered_observations)

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
            transformed_detections_path = os.path.join(
                test_orbit_directory, "transformed_detections.parquet"
            )
            transformed_detections.to_parquet(transformed_detections_path)

        yield LinkTestOrbitStageResult(
            name="range_and_transform",
            result=(transformed_detections,),
            path=(transformed_detections_path,),
        )

        checkpoint = CheckpointData(
            stage="cluster_and_link",
            filtered_observations=filtered_observations,
            transformed_detections=transformed_detections,
        )

    # TODO: ray support for the rest of the pipeline has not yet been implemented
    # so we will convert the ray objects to regular objects for now
    if use_ray:
        if isinstance(checkpoint.filtered_observations, ray.ObjectRef):
            checkpoint.filtered_observations = ray.get(filtered_observations)

    if checkpoint.stage == "cluster_and_link":
        transformed_detections = checkpoint.transformed_detections
        filtered_observations = checkpoint.filtered_observations
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
            cluster_members_path = os.path.join(
                test_orbit_directory, "cluster_members.parquet"
            )
            clusters.to_parquet(clusters_path)
            cluster_members.to_parquet(cluster_members_path)

        yield LinkTestOrbitStageResult(
            name="cluster_and_link",
            result=(clusters, cluster_members),
            path=(clusters_path, cluster_members_path),
        )

        checkpoint = CheckpointData(
            stage="initial_orbit_determination",
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
            iod_orbit_members_path = os.path.join(
                test_orbit_directory, "iod_orbit_members.parquet"
            )
            iod_orbits.to_parquet(iod_orbits_path)
            iod_orbit_members.to_parquet(iod_orbit_members_path)

        yield LinkTestOrbitStageResult(
            name="initial_orbit_determination",
            result=(iod_orbits, iod_orbit_members),
            path=(iod_orbits_path, iod_orbit_members_path),
        )

        checkpoint = CheckpointData(
            stage="differential_correction",
            filtered_observations=filtered_observations,
            iod_orbits=iod_orbits,
            iod_orbit_members=iod_orbit_members,
        )

    if checkpoint.stage == "differential_correction":
        iod_orbits = checkpoint.iod_orbits
        iod_orbit_members = checkpoint.iod_orbit_members
        filtered_observations = checkpoint.filtered_observations
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

        od_orbits_path = None
        od_orbit_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving OD orbits to {test_orbit_directory}...")
            od_orbits_path = os.path.join(test_orbit_directory, "od_orbits.parquet")
            od_orbit_members_path = os.path.join(
                test_orbit_directory, "od_orbit_members.parquet"
            )
            od_orbits.to_parquet(od_orbits_path)
            od_orbit_members.to_parquet(od_orbit_members_path)

        yield LinkTestOrbitStageResult(
            name="differential_correction",
            result=(od_orbits, od_orbit_members),
            path=(od_orbits_path, od_orbit_members_path),
        )

        checkpoint = CheckpointData(
            stage="recover_orbits",
            filtered_observations=filtered_observations,
            od_orbits=od_orbits,
            od_orbit_members=od_orbit_members,
        )

    if checkpoint.stage == "recover_orbits":
        od_orbits = checkpoint.od_orbits
        od_orbit_members = checkpoint.od_orbit_members
        filtered_observations = checkpoint.filtered_observations
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

        recovered_orbits_path = None
        recovered_orbit_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving recovered orbits to {test_orbit_directory}...")
            recovered_orbits_path = os.path.join(
                test_orbit_directory, "recovered_orbits.parquet"
            )
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
