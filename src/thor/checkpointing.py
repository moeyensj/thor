import logging
import pathlib
from typing import Annotated, Dict, Literal, Optional, Type, Union

import pydantic
import quivr as qv
import ray

from thor.clusters import ClusterMembers, Clusters
from thor.observations.observations import Observations
from thor.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from thor.range_and_transform import TransformedDetections

logger = logging.getLogger("thor")


VALID_STAGES = Literal[
    "filter_observations",
    "range_and_transform",
    "cluster_and_link",
    "initial_orbit_determination",
    "differential_correction",
    "recover_orbits",
    "complete",
]


class FilterObservations(pydantic.BaseModel):
    stage: Literal["filter_observations"]


class RangeAndTransform(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["range_and_transform"]
    filtered_observations: Union[Observations, ray.ObjectRef]


class ClusterAndLink(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["cluster_and_link"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    transformed_detections: Union[TransformedDetections, ray.ObjectRef]


class InitialOrbitDetermination(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["initial_orbit_determination"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    clusters: Union[Clusters, ray.ObjectRef]
    cluster_members: Union[ClusterMembers, ray.ObjectRef]


class DifferentialCorrection(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["differential_correction"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    iod_orbits: Union[FittedOrbits, ray.ObjectRef]
    iod_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


class RecoverOrbits(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["recover_orbits"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    od_orbits: Union[FittedOrbits, ray.ObjectRef]
    od_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


class Complete(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    stage: Literal["complete"]
    recovered_orbits: Union[FittedOrbits, ray.ObjectRef]
    recovered_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


CheckpointData = Annotated[
    Union[
        FilterObservations,
        RangeAndTransform,
        ClusterAndLink,
        InitialOrbitDetermination,
        DifferentialCorrection,
        RecoverOrbits,
        Complete,
    ],
    pydantic.Field(discriminator="stage"),
]

# A mapping from stage to model class
stage_to_model: Dict[str, Type[pydantic.BaseModel]] = {
    "filter_observations": FilterObservations,
    "range_and_transform": RangeAndTransform,
    "cluster_and_link": ClusterAndLink,
    "initial_orbit_determination": InitialOrbitDetermination,
    "differential_correction": DifferentialCorrection,
    "recover_orbits": RecoverOrbits,
    "complete": Complete,
}


def create_checkpoint_data(stage: VALID_STAGES, **data) -> CheckpointData:
    """
    Create checkpoint data from the given stage and data.
    """
    model = stage_to_model.get(stage)
    if model:
        return model(stage=stage, **data)
    raise ValueError(f"Invalid stage: {stage}")


def detect_checkpoint_stage(test_orbit_directory: pathlib.Path) -> VALID_STAGES:
    """
    Looks for existing files and indicates the next stage to run
    """
    if not test_orbit_directory.is_dir():
        raise ValueError(f"{test_orbit_directory} is not a directory")

    if not test_orbit_directory.exists():
        logger.info("Working directory does not exist, starting at beginning.")
        return "filter_observations"

    if not (test_orbit_directory / "filtered_observations.parquet").exists():
        logger.info("No filtered observations found, starting stage filter_observations")
        return "filter_observations"

    if (test_orbit_directory / "recovered_orbits.parquet").exists() and (
        test_orbit_directory / "recovered_orbit_members.parquet"
    ).exists():
        logger.info("Found recovered orbits, pipeline is complete.")
        return "complete"

    if (test_orbit_directory / "od_orbits.parquet").exists() and (
        test_orbit_directory / "od_orbit_members.parquet"
    ).exists():
        logger.info("Found OD orbits, starting stage recover_orbits")
        return "recover_orbits"

    if (test_orbit_directory / "iod_orbits.parquet").exists() and (
        test_orbit_directory / "iod_orbit_members.parquet"
    ).exists():
        logger.info("Found IOD orbits, starting stage differential_correction")
        return "differential_correction"

    if (test_orbit_directory / "clusters.parquet").exists() and (
        test_orbit_directory / "cluster_members.parquet"
    ).exists():
        logger.info("Found clusters, starting stage initial_orbit_determination")
        return "initial_orbit_determination"

    if (test_orbit_directory / "transformed_detections.parquet").exists():
        logger.info("Found transformed detections, starting stage cluster_and_link")
        return "cluster_and_link"

    if (test_orbit_directory / "filtered_observations.parquet").exists():
        logger.info("Found filtered observations, starting stage range_and_transform")
        return "range_and_transform"

    raise ValueError(f"Could not detect stage from {test_orbit_directory}")


def load_initial_checkpoint_values(
    test_orbit_directory: Optional[Union[pathlib.Path, str]] = None,
) -> CheckpointData:
    """
    Check for completed stages and return values from disk if they exist.

    We want to avoid loading objects into memory that are not required.
    """
    if isinstance(test_orbit_directory, str):
        test_orbit_directory = pathlib.Path(test_orbit_directory)

    if test_orbit_directory is None:
        logger.info("Not using a workign directory, start at beginning.")
        return create_checkpoint_data("filter_observations")

    stage: VALID_STAGES = detect_checkpoint_stage(test_orbit_directory)

    if stage == "filter_observations":
        return create_checkpoint_data(stage)

    # If we've already completed the pipeline, we can load the recovered orbits
    if stage == "complete":
        recovered_orbits_path = pathlib.Path(test_orbit_directory, "recovered_orbits.parquet")
        recovered_orbit_members_path = pathlib.Path(test_orbit_directory, "recovered_orbit_members.parquet")
        recovered_orbits = FittedOrbits.from_parquet(recovered_orbits_path)
        recovered_orbit_members = FittedOrbitMembers.from_parquet(recovered_orbit_members_path)

        if recovered_orbits.fragmented():
            recovered_orbits = qv.defragment(recovered_orbits)
        if recovered_orbit_members.fragmented():
            recovered_orbit_members = qv.defragment(recovered_orbit_members)

        return create_checkpoint_data(
            stage,
            recovered_orbits=recovered_orbits,
            recovered_orbit_members=recovered_orbit_members,
        )

    # Filtered observations are required for all other stages
    filtered_observations_path = pathlib.Path(test_orbit_directory, "filtered_observations.parquet")

    filtered_observations = Observations.from_parquet(filtered_observations_path)
    if filtered_observations.fragmented():
        filtered_observations = qv.defragment(filtered_observations)

    if stage == "recover_orbits":
        od_orbits_path = pathlib.Path(test_orbit_directory, "od_orbits.parquet")
        od_orbit_members_path = pathlib.Path(test_orbit_directory, "od_orbit_members.parquet")
        if od_orbits_path.exists() and od_orbit_members_path.exists():
            logger.info("Found OD orbits in checkpoint")
            od_orbits = FittedOrbits.from_parquet(od_orbits_path)
            od_orbit_members = FittedOrbitMembers.from_parquet(od_orbit_members_path)

            if od_orbits.fragmented():
                od_orbits = qv.defragment(od_orbits)
            if od_orbit_members.fragmented():
                od_orbit_members = qv.defragment(od_orbit_members)

            return create_checkpoint_data(
                "recover_orbits",
                filtered_observations=filtered_observations,
                od_orbits=od_orbits,
                od_orbit_members=od_orbit_members,
            )

    if stage == "differential_correction":
        iod_orbits_path = pathlib.Path(test_orbit_directory, "iod_orbits.parquet")
        iod_orbit_members_path = pathlib.Path(test_orbit_directory, "iod_orbit_members.parquet")
        if iod_orbits_path.exists() and iod_orbit_members_path.exists():
            logger.info("Found IOD orbits")
            iod_orbits = FittedOrbits.from_parquet(iod_orbits_path)
            iod_orbit_members = FittedOrbitMembers.from_parquet(iod_orbit_members_path)

            if iod_orbits.fragmented():
                iod_orbits = qv.defragment(iod_orbits)
            if iod_orbit_members.fragmented():
                iod_orbit_members = qv.defragment(iod_orbit_members)

            return create_checkpoint_data(
                "differential_correction",
                filtered_observations=filtered_observations,
                iod_orbits=iod_orbits,
                iod_orbit_members=iod_orbit_members,
            )

    if stage == "initial_orbit_determination":
        clusters_path = pathlib.Path(test_orbit_directory, "clusters.parquet")
        cluster_members_path = pathlib.Path(test_orbit_directory, "cluster_members.parquet")
        if clusters_path.exists() and cluster_members_path.exists():
            logger.info("Found clusters")
            clusters = Clusters.from_parquet(clusters_path)
            cluster_members = ClusterMembers.from_parquet(cluster_members_path)

            if clusters.fragmented():
                clusters = qv.defragment(clusters)
            if cluster_members.fragmented():
                cluster_members = qv.defragment(cluster_members)

            return create_checkpoint_data(
                "initial_orbit_determination",
                filtered_observations=filtered_observations,
                clusters=clusters,
                cluster_members=cluster_members,
            )

    if stage == "cluster_and_link":
        transformed_detections_path = pathlib.Path(test_orbit_directory, "transformed_detections.parquet")
        if transformed_detections_path.exists():
            logger.info("Found transformed detections")
            transformed_detections = TransformedDetections.from_parquet(transformed_detections_path)

            return create_checkpoint_data(
                "cluster_and_link",
                filtered_observations=filtered_observations,
                transformed_detections=transformed_detections,
            )

    return create_checkpoint_data("range_and_transform", filtered_observations=filtered_observations)
