import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import pyarrow.parquet as pq
import quivr as qv
import ray

from thor.clusters import ClusterMembers, Clusters, FittedClusterMembers, FittedClusters
from thor.observations.observations import Observations
from thor.orbit import TestOrbitEphemeris, TestOrbits
from thor.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from thor.range_and_transform import TransformedDetections

logger = logging.getLogger("thor")


VALID_STAGES = Literal[
    "filter_observations",
    "generate_ephemeris",
    "range_and_transform",
    "form_tracklets",
    "cluster_and_link",
    "fit_clusters",
    "initial_orbit_determination",
    "differential_correction",
    "recover_orbits",
    "complete",
]


@dataclass
class FilterObservations:
    stage: Literal["filter_observations"]


@dataclass
class GenerateEphemeris:
    stage: Literal["generate_ephemeris"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    test_orbit_ephemeris: Union[TestOrbitEphemeris, ray.ObjectRef]


@dataclass
class RangeAndTransform:
    stage: Literal["range_and_transform"]
    test_orbit_ephemeris: Union[TestOrbitEphemeris, ray.ObjectRef]
    filtered_observations: Union[Observations, ray.ObjectRef]


@dataclass
class FormTracklets:
    stage: Literal["form_tracklets"]
    test_orbit_ephemeris: Union[TestOrbitEphemeris, ray.ObjectRef]
    filtered_observations: Union[Observations, ray.ObjectRef]
    transformed_detections: Union[TransformedDetections, ray.ObjectRef]


@dataclass
class ClusterAndLink:
    stage: Literal["cluster_and_link"]
    test_orbit_ephemeris: Union[TestOrbitEphemeris, ray.ObjectRef]
    filtered_observations: Union[Observations, ray.ObjectRef]
    transformed_detections: Union[TransformedDetections, ray.ObjectRef]
    tracklets: Optional[object] = None
    tracklet_members: Optional[object] = None


@dataclass
class FitClusters:
    stage: Literal["fit_clusters"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    clusters: Union[Clusters, ray.ObjectRef]
    cluster_members: Union[ClusterMembers, ray.ObjectRef]
    transformed_detections: Union[TransformedDetections, ray.ObjectRef]


@dataclass
class InitialOrbitDetermination:
    stage: Literal["initial_orbit_determination"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    clusters: Union[FittedClusters, ray.ObjectRef]
    cluster_members: Union[FittedClusterMembers, ray.ObjectRef]


@dataclass
class DifferentialCorrection:
    stage: Literal["differential_correction"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    iod_orbits: Union[FittedOrbits, ray.ObjectRef]
    iod_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


@dataclass
class RecoverOrbits:
    stage: Literal["recover_orbits"]
    filtered_observations: Union[Observations, ray.ObjectRef]
    od_orbits: Union[FittedOrbits, ray.ObjectRef]
    od_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


@dataclass
class Complete:
    stage: Literal["complete"]
    recovered_orbits: Union[FittedOrbits, ray.ObjectRef]
    recovered_orbit_members: Union[FittedOrbitMembers, ray.ObjectRef]


CheckpointData = Union[
    FilterObservations,
    GenerateEphemeris,
    RangeAndTransform,
    FormTracklets,
    ClusterAndLink,
    FitClusters,
    InitialOrbitDetermination,
    DifferentialCorrection,
    RecoverOrbits,
    Complete,
]

# A mapping from stage to model class
stage_to_model: Dict[str, type] = {
    "filter_observations": FilterObservations,
    "generate_ephemeris": GenerateEphemeris,
    "range_and_transform": RangeAndTransform,
    "form_tracklets": FormTracklets,
    "cluster_and_link": ClusterAndLink,
    "fit_clusters": FitClusters,
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

    if not (test_orbit_directory / "test_orbit_ephemeris.parquet").exists():
        logger.info("No test orbit ephemeris found, starting stage generate_ephemeris")
        return "generate_ephemeris"

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

    if (test_orbit_directory / "fitted_clusters.parquet").exists() and (
        test_orbit_directory / "fitted_cluster_members.parquet"
    ).exists():
        logger.info("Found fitted clusters, starting stage initial_orbit_determination")
        return "initial_orbit_determination"

    if (test_orbit_directory / "clusters.parquet").exists() and (
        test_orbit_directory / "cluster_members.parquet"
    ).exists():
        logger.info("Found unfitted clusters, starting stage fit_clusters")
        return "fit_clusters"

    if (test_orbit_directory / "tracklets.parquet").exists() and (
        test_orbit_directory / "tracklet_members.parquet"
    ).exists():
        logger.info("Found tracklets, starting stage cluster_and_link")
        return "cluster_and_link"

    if (test_orbit_directory / "transformed_detections.parquet").exists():
        logger.info("Found transformed detections, starting stage form_tracklets")
        return "form_tracklets"

    if (test_orbit_directory / "test_orbit_ephemeris.parquet").exists():
        logger.info("Found test orbit ephemeris, starting stage range_and_transform")
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
        logger.info("Not using a working directory, start at beginning.")
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

    # Handle generate_ephemeris stage (needs filtered_observations, ephemeris may be empty)
    if stage == "generate_ephemeris":
        # Return empty ephemeris - it will be generated in the stage
        return create_checkpoint_data(
            stage,
            filtered_observations=filtered_observations,
            test_orbit_ephemeris=TestOrbitEphemeris.empty(),
        )

    # Test orbit ephemeris is required for range_and_transform and cluster_and_link
    test_orbit_ephemeris_path = pathlib.Path(test_orbit_directory, "test_orbit_ephemeris.parquet")
    test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(test_orbit_ephemeris_path)
    if test_orbit_ephemeris.fragmented():
        test_orbit_ephemeris = qv.defragment(test_orbit_ephemeris)

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
        fitted_clusters_path = pathlib.Path(test_orbit_directory, "fitted_clusters.parquet")
        fitted_cluster_members_path = pathlib.Path(test_orbit_directory, "fitted_cluster_members.parquet")
        if fitted_clusters_path.exists() and fitted_cluster_members_path.exists():
            logger.info("Found fitted clusters")
            clusters = FittedClusters.from_parquet(fitted_clusters_path)
            cluster_members = FittedClusterMembers.from_parquet(fitted_cluster_members_path)

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

    if stage == "fit_clusters":
        clusters_path = pathlib.Path(test_orbit_directory, "clusters.parquet")
        cluster_members_path = pathlib.Path(test_orbit_directory, "cluster_members.parquet")
        transformed_detections_path = pathlib.Path(test_orbit_directory, "transformed_detections.parquet")
        if clusters_path.exists() and cluster_members_path.exists():
            logger.info("Found unfitted clusters")
            clusters = Clusters.from_parquet(clusters_path)
            cluster_members = ClusterMembers.from_parquet(cluster_members_path)
            transformed_detections = TransformedDetections.from_parquet(transformed_detections_path)

            if clusters.fragmented():
                clusters = qv.defragment(clusters)
            if cluster_members.fragmented():
                cluster_members = qv.defragment(cluster_members)

            return create_checkpoint_data(
                "fit_clusters",
                filtered_observations=filtered_observations,
                clusters=clusters,
                cluster_members=cluster_members,
                transformed_detections=transformed_detections,
            )

    if stage == "cluster_and_link":
        transformed_detections_path = pathlib.Path(test_orbit_directory, "transformed_detections.parquet")
        tracklets_path = pathlib.Path(test_orbit_directory, "tracklets.parquet")
        tracklet_members_path = pathlib.Path(test_orbit_directory, "tracklet_members.parquet")

        if transformed_detections_path.exists():
            logger.info("Found transformed detections")
            transformed_detections = TransformedDetections.from_parquet(transformed_detections_path)

            tracklets = None
            tracklet_members_data = None
            if tracklets_path.exists() and tracklet_members_path.exists():
                from thor.clustering.tracklets import TrackletMembers as TM
                from thor.clustering.tracklets import Tracklets as T

                logger.info("Found tracklets")
                tracklets = T.from_parquet(tracklets_path)
                tracklet_members_data = TM.from_parquet(tracklet_members_path)

            return create_checkpoint_data(
                "cluster_and_link",
                test_orbit_ephemeris=test_orbit_ephemeris,
                filtered_observations=filtered_observations,
                transformed_detections=transformed_detections,
                tracklets=tracklets,
                tracklet_members=tracklet_members_data,
            )

    if stage == "form_tracklets":
        transformed_detections_path = pathlib.Path(test_orbit_directory, "transformed_detections.parquet")
        if transformed_detections_path.exists():
            logger.info("Found transformed detections, starting tracklet formation")
            transformed_detections = TransformedDetections.from_parquet(transformed_detections_path)

            return create_checkpoint_data(
                "form_tracklets",
                test_orbit_ephemeris=test_orbit_ephemeris,
                filtered_observations=filtered_observations,
                transformed_detections=transformed_detections,
            )

    return create_checkpoint_data(
        "range_and_transform",
        test_orbit_ephemeris=test_orbit_ephemeris,
        filtered_observations=filtered_observations,
    )


# Mapping of output keys to (filename, loader class)
_OUTPUT_FILES = {
    "test_orbit_ephemeris": ("test_orbit_ephemeris.parquet", TestOrbitEphemeris),
    "filtered_observations": ("filtered_observations.parquet", Observations),
    "transformed_detections": ("transformed_detections.parquet", TransformedDetections),
    "clusters": ("clusters.parquet", Clusters),
    "cluster_members": ("cluster_members.parquet", ClusterMembers),
    "fitted_clusters": ("fitted_clusters.parquet", FittedClusters),
    "fitted_cluster_members": ("fitted_cluster_members.parquet", FittedClusterMembers),
    "iod_orbits": ("iod_orbits.parquet", FittedOrbits),
    "iod_orbit_members": ("iod_orbit_members.parquet", FittedOrbitMembers),
    "od_orbits": ("od_orbits.parquet", FittedOrbits),
    "od_orbit_members": ("od_orbit_members.parquet", FittedOrbitMembers),
    "recovered_orbits": ("recovered_orbits.parquet", FittedOrbits),
    "recovered_orbit_members": ("recovered_orbit_members.parquet", FittedOrbitMembers),
}


class RunResults(qv.Table):
    """
    A per-orbit index of persisted pipeline outputs (paths only).

    One row corresponds to one `test_orbit_id` (typically the orbit directory name).
    Path columns are stored as strings (nullable). This table does *not* load any
    results into memory; use `load_orbit()` to materialize the objects for a single orbit.
    """

    test_orbit_id = qv.LargeStringColumn()
    child_of = qv.LargeStringColumn(nullable=True)

    test_orbit_ephemeris = qv.Int64Column(default=0)
    filtered_observations = qv.Int64Column(default=0)
    transformed_detections = qv.Int64Column(default=0)
    clusters = qv.Int64Column(default=0)
    cluster_members = qv.Int64Column(default=0)
    fitted_clusters = qv.Int64Column(default=0)
    fitted_cluster_members = qv.Int64Column(default=0)
    iod_orbits = qv.Int64Column(default=0)
    iod_orbit_members = qv.Int64Column(default=0)
    od_orbits = qv.Int64Column(default=0)
    od_orbit_members = qv.Int64Column(default=0)
    recovered_orbits = qv.Int64Column(default=0)
    recovered_orbit_members = qv.Int64Column(default=0)

    test_orbit_file = qv.LargeStringColumn(nullable=True)
    test_orbit_ephemeris_file = qv.LargeStringColumn(nullable=True)
    filtered_observations_file = qv.LargeStringColumn(nullable=True)
    transformed_detections_file = qv.LargeStringColumn(nullable=True)
    clusters_file = qv.LargeStringColumn(nullable=True)
    cluster_members_file = qv.LargeStringColumn(nullable=True)
    fitted_clusters_file = qv.LargeStringColumn(nullable=True)
    fitted_cluster_members_file = qv.LargeStringColumn(nullable=True)
    iod_orbits_file = qv.LargeStringColumn(nullable=True)
    iod_orbit_members_file = qv.LargeStringColumn(nullable=True)
    od_orbits_file = qv.LargeStringColumn(nullable=True)
    od_orbit_members_file = qv.LargeStringColumn(nullable=True)
    recovered_orbits_file = qv.LargeStringColumn(nullable=True)
    recovered_orbit_members_file = qv.LargeStringColumn(nullable=True)

    run_dir = qv.StringAttribute()

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Union[str, pathlib.Path],
        store_relative_paths: bool = True,
    ) -> "RunResults":
        """
        Build a `RunResults` index from a run directory.

        Parameters
        ----------
        run_dir: str
            Run directory (e.g. `orbit-dev-07`). This function will also detect any nested
            run directories under this directory (identified by `inputs/config.json`) and
            include their orbits as well.
        store_relative_paths: bool
            If True, store paths relative to `run_dir`. If False, store absolute paths.
        """
        root = pathlib.Path(run_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if not root.is_dir():
            raise NotADirectoryError(root)

        # Names directly under the provided run directory. For nested "split runs"
        # (e.g. <run_dir>/<orbit_id>/<split_run>/inputs/config.json), the parent orbit id
        # is inferred as the first path component under `root`, even if that directory
        # itself is also a run directory (i.e. it contains an `inputs/`).
        root_children = {
            d.name for d in root.iterdir() if d.is_dir() and d.name != "inputs" and not d.name.startswith(".")
        }

        # Determine run directories from the on-disk structure.
        run_dirs: List[pathlib.Path] = []
        if (root / "inputs").is_dir():
            run_dirs.append(root)
        for config_path in root.rglob("inputs/config.json"):
            run_dirs.append(config_path.parent.parent)

        # De-duplicate while preserving a stable order.
        run_dirs = sorted({p.resolve() for p in run_dirs}, key=lambda p: str(p))

        # Column builders
        test_orbit_ids: List[str] = []
        child_ofs: List[str] = []
        cols_files: Dict[str, List[Optional[str]]] = {
            "test_orbit_file": [],
            "test_orbit_ephemeris_file": [],
            "filtered_observations_file": [],
            "transformed_detections_file": [],
            "clusters_file": [],
            "cluster_members_file": [],
            "fitted_clusters_file": [],
            "fitted_cluster_members_file": [],
            "iod_orbits_file": [],
            "iod_orbit_members_file": [],
            "od_orbits_file": [],
            "od_orbit_members_file": [],
            "recovered_orbits_file": [],
            "recovered_orbit_members_file": [],
        }
        cols_counts: Dict[str, List[Optional[int]]] = {
            "test_orbit_ephemeris": [],
            "filtered_observations": [],
            "transformed_detections": [],
            "clusters": [],
            "cluster_members": [],
            "fitted_clusters": [],
            "fitted_cluster_members": [],
            "iod_orbits": [],
            "iod_orbit_members": [],
            "od_orbits": [],
            "od_orbit_members": [],
            "recovered_orbits": [],
            "recovered_orbit_members": [],
        }

        def format_path(p: pathlib.Path) -> str:
            return str(p.relative_to(root) if store_relative_paths else p.resolve())

        def row_count(p: pathlib.Path) -> int:
            return int(pq.ParquetFile(p).metadata.num_rows)

        for run_path in run_dirs:
            # Infer `child_of` from directory structure:
            # - for the root run directory, there is no parent orbit id
            # - for nested runs, use the first path component under root
            parent_orbit_id: Optional[str] = None
            if run_path != root:
                rel = run_path.relative_to(root)
                if len(rel.parts) > 0 and rel.parts[0] in root_children:
                    parent_orbit_id = rel.parts[0]

            # Orbit ids are defined by inputs/<orbit_id>/... (authoritative)
            inputs_dir = run_path / "inputs"
            orbit_ids: List[str] = []
            if inputs_dir.is_dir():
                orbit_ids = sorted(
                    [d.name for d in inputs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
                )

            for orbit_id in orbit_ids:
                test_orbit_ids.append(orbit_id)
                child_ofs.append(parent_orbit_id)

                # Inputs: <run_dir>/inputs/<orbit_id>/test_orbit.parquet
                test_orbit_path = inputs_dir / orbit_id / "test_orbit.parquet"
                if test_orbit_path.exists():
                    cols_files["test_orbit_file"].append(format_path(test_orbit_path))
                else:
                    cols_files["test_orbit_file"].append(None)

                # Outputs: <run_dir>/<orbit_id>/<filename>
                orbit_out_dir = run_path / orbit_id
                for key, (filename, _loader) in _OUTPUT_FILES.items():
                    path = orbit_out_dir / filename
                    file_col = f"{key}_file"
                    if path.exists():
                        cols_files[file_col].append(format_path(path))
                        cols_counts[key].append(row_count(path))
                    else:
                        cols_files[file_col].append(None)
                        cols_counts[key].append(0)

        return cls.from_kwargs(
            run_dir=str(root),
            test_orbit_id=test_orbit_ids,
            child_of=child_ofs,
            **cols_files,
            **cols_counts,
        )

    def load_orbit(
        self,
        test_orbit_id: str,
        run_dir: Optional[Union[str, pathlib.Path]] = None,
        defragment: bool = True,
    ) -> Dict[str, Optional[qv.AnyTable]]:
        """
        Load all available results for a single `test_orbit_id` into memory.

        Parameters
        ----------
        test_orbit_id: str
            Orbit directory name to load.
        run_dir: str
            Base directory for resolving relative paths. Required if this table stores
            relative paths.
        defragment: bool
            If True, defragment loaded tables when needed.

        Returns
        -------
        orbit_results: Dict[str, Optional[qv.AnyTable]]
            Mapping from output key (e.g. "clusters") to the loaded table (or None).
        """
        row = self.select("test_orbit_id", test_orbit_id)
        if len(row) == 0:
            raise KeyError(f"test_orbit_id not found: {test_orbit_id}")

        # If duplicates exist, prefer the first deterministically.
        row0 = row.take([0])

        base = pathlib.Path(run_dir) if run_dir is not None else pathlib.Path(self.run_dir)

        def _resolve(path_str: str, col_name: str) -> pathlib.Path:
            path = pathlib.Path(path_str)
            if not path.is_absolute():
                if base is None:
                    raise ValueError(
                        f"RunResults stores relative paths; pass run_dir to load_orbit() "
                        f"(missing for {test_orbit_id}, {col_name}={path_str})."
                    )
                path = base / path
            return path

        loaded: Dict[str, Optional[qv.AnyTable]] = {}

        test_orbit_file = row0.test_orbit_file.to_pylist()[0]
        if test_orbit_file is None:
            loaded["test_orbit"] = TestOrbits.empty()

        else:
            path = _resolve(test_orbit_file, "test_orbit_file")
            obj = TestOrbits.from_parquet(path)
            if defragment and obj.fragmented():
                obj = qv.defragment(obj)
            loaded["test_orbit"] = obj

        # Outputs
        for key, (_filename, loader) in _OUTPUT_FILES.items():
            path_str = getattr(row0, f"{key}_file").to_pylist()[0]
            if path_str is None:
                loaded[key] = loader.empty()
                continue
            path = _resolve(path_str, f"{key}_file")
            obj = loader.from_parquet(path)
            if defragment and obj.fragmented():
                obj = qv.defragment(obj)
            loaded[key] = obj

        return loaded
