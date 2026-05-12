import importlib
import logging
import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import pyarrow.parquet as pq
import quivr as qv

from .checkpointing import detect_checkpoint_stage
from .clusters import FittedClusterMembers, FittedClusters, cluster_and_link
from .config import Config, initialize_config
from .observations.filters import (
    ObservationFilter,
    TestOrbitMahalanobisObservationFilter,
    TestOrbitRadiusObservationFilter,
    filter_observations,
)
from .observations.observations import Observations, get_observation_times
from .orbit import TestOrbitEphemeris, TestOrbits
from .orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from .orbits import (
    differential_correction,
    initial_orbit_determination,
    merge_and_extend_orbits,
)
from .range_and_transform import TransformedDetections, range_and_transform

logger = logging.getLogger("thor")


def _log_parquet_metadata(path: str, context: str) -> None:
    """
    Log the metadata of a parquet file for debugging.
    """
    log = logging.getLogger("thor")
    try:
        schema = pq.read_schema(path)
        metadata = schema.metadata or {}
        time_scale = metadata.get(b"coordinates.time.scale")
        frame = metadata.get(b"coordinates.frame")
        log.info(
            "Parquet metadata (%s): %s - time.scale=%s, frame=%s",
            context,
            path,
            time_scale.decode("utf-8") if time_scale else None,
            frame.decode("utf-8") if frame else None,
        )
    except Exception as e:
        log.info("Failed to read parquet metadata for %s (%s): %s", path, context, e)



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
        "generate_ephemeris",
        "range_and_transform",
        "cluster_and_link",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
    ]
    # The orbit_id for the specific test orbit that produced this stage result.
    #
    # With recursive test-orbit splitting, a single call to `link_test_orbits(...)`
    # can execute a root orbit and multiple child orbits. Consumers must not
    # assume a stage result belongs to the "root" orbit.
    test_orbit_id: str
    # Paths to stage outputs on disk. This is the *only* data payload passed
    # between stages and across process boundaries.
    #
    # Notes:
    # - Paths may be None when running without a working directory (in-memory
    #   mode). The pipeline/orchestrator uses a working directory, so paths are
    #   always present there.
    # - We intentionally do NOT store in-memory Quivr tables here to avoid
    #   accidentally retaining huge intermediates across recursive splitting.
    path: Tuple[Optional[str], ...] = (None,)
    # Lightweight per-stage summary information (counts, etc.). This is used by
    # downstream consumers (e.g., the pipeline remote runner) for metrics and
    # reporting.
    stats: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize types for downstream consumers. Many call sites already pass
        # tuples, but we keep this robust and future-proof.
        if not isinstance(self.path, tuple):
            self.path = tuple(self.path)
        if self.stats is None:
            self.stats = {}
        if not isinstance(self.test_orbit_id, str):
            self.test_orbit_id = str(self.test_orbit_id)


def link_test_orbit(
    test_orbit: TestOrbits,
    observations: Union[str, Observations],
    working_dir: Optional[str] = None,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
    use_orbit_subdir: bool = True,
) -> Iterator[LinkTestOrbitStageResult]:
    """
    Run THOR for a single test orbit on the given observations. This function will yield
    results at each stage of the pipeline.
        1. filtered_observations
        2. test_orbit_ephemeris
        3. transformed_detections
        4. clusters, cluster_members
        5. iod_orbits, iod_orbit_members
        6. od_orbits, od_orbit_members
        7. recovered_orbits, recovered_orbit_members

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
    use_orbit_subdir : bool, optional
        Whether to use a subdirectory for the test orbit. If True, the test orbit will be stored in a subdirectory of the working directory.
        If False, the test orbit will be stored in the working directory.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    if len(test_orbit) != 1:
        raise ValueError(f"link_test_orbit received {len(test_orbit)} orbits but expected 1.")
    test_orbit_id = test_orbit.orbit_id[0].as_py()

    test_orbit_directory = None
    if working_dir is not None:
        working_dir_path = pathlib.Path(working_dir)
        logger.info(f"Using working directory: {working_dir}")
        test_orbit_directory = (
            pathlib.Path(working_dir, test_orbit.orbit_id[0].as_py())
            if use_orbit_subdir
            else working_dir_path
        )
        test_orbit_directory.mkdir(parents=True, exist_ok=True)
        inputs_dir = pathlib.Path(working_dir_path, "inputs")
        inputs_dir.mkdir(parents=True, exist_ok=True)

    initialize_test_orbit(test_orbit, test_orbit_directory if not use_orbit_subdir else working_dir)

    if config is None:
        config = Config()

    stop_after_stage = config.stop_after_stage

    initialize_config(config, working_dir)

    module_path, class_name = config.propagator_namespace.rsplit(".", 1)
    propagator_module = importlib.import_module(module_path)
    propagator_class = getattr(propagator_module, class_name)

    # ------------------------------------------------------------------
    # Filesystem-based stage resume
    # ------------------------------------------------------------------
    #
    # We intentionally avoid "checkpoint objects" that hold in-memory Quivr
    # tables or Ray ObjectRefs. The filesystem is the source of truth: we detect
    # which stage to run next based on the presence of stage output Parquet
    # files under the per-orbit directory.
    if test_orbit_directory is None:
        next_stage: str = "filter_observations"
    else:
        next_stage = detect_checkpoint_stage(pathlib.Path(test_orbit_directory))
    logger.info("Starting at stage: %s", next_stage)

    def _p(name: str) -> Optional[str]:
        if test_orbit_directory is None:
            return None
        return os.path.join(test_orbit_directory, name)

    # ------------------------------------------------------------------
    # Respect stop_after_stage even when resuming from disk
    # ------------------------------------------------------------------
    #
    # `detect_checkpoint_stage(...)` can return a stage *after* stop_after_stage
    # when the working directory already contains later-stage outputs (e.g. from
    # a previous full run). In that situation we must NOT proceed into later
    # stages: stop_after_stage is a hard upper bound on executed work.
    _STAGE_ORDER: list[str] = [
        "filter_observations",
        "generate_ephemeris",
        "range_and_transform",
        "cluster_and_link",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
    ]

    def _stage_rank(stage: Optional[str]) -> Optional[int]:
        if stage is None:
            return None
        s = str(stage).strip()
        if not s:
            return None
        if s == "complete":
            return len(_STAGE_ORDER)
        try:
            return _STAGE_ORDER.index(s)
        except ValueError:
            return None

    def _checkpoint_stage_result(stage: str) -> Optional[LinkTestOrbitStageResult]:
        if test_orbit_directory is None:
            return None
        s = str(stage).strip()
        if not s:
            return None
        try:
            if s == "filter_observations":
                fo_path = _p("filtered_observations.parquet")
                if fo_path is None or not pathlib.Path(fo_path).is_file():
                    return None
                n = int(pq.ParquetFile(fo_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="filter_observations",
                    test_orbit_id=test_orbit_id,
                    path=(fo_path,),
                    stats={"num_filtered_observations": n},
                )
            if s == "generate_ephemeris":
                eph_path = _p("test_orbit_ephemeris.parquet")
                if eph_path is None or not pathlib.Path(eph_path).is_file():
                    return None
                n = int(pq.ParquetFile(eph_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="generate_ephemeris",
                    test_orbit_id=test_orbit_id,
                    path=(eph_path,),
                    stats={"num_test_orbit_ephemeris_rows": n},
                )
            if s == "range_and_transform":
                td_path = _p("transformed_detections.parquet")
                if td_path is None or not pathlib.Path(td_path).is_file():
                    return None
                n = int(pq.ParquetFile(td_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="range_and_transform",
                    test_orbit_id=test_orbit_id,
                    path=(td_path,),
                    stats={"num_transformed_detections": n},
                )
            if s == "cluster_and_link":
                c_path = _p("clusters.parquet")
                m_path = _p("cluster_members.parquet")
                if (
                    c_path is None
                    or m_path is None
                    or not pathlib.Path(c_path).is_file()
                    or not pathlib.Path(m_path).is_file()
                ):
                    return None
                n_c = int(pq.ParquetFile(c_path).metadata.num_rows)
                n_m = int(pq.ParquetFile(m_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="cluster_and_link",
                    test_orbit_id=test_orbit_id,
                    path=(c_path, m_path),
                    stats={"num_clusters": n_c, "num_cluster_members": n_m},
                )
            if s == "initial_orbit_determination":
                o_path = _p("iod_orbits.parquet")
                m_path = _p("iod_orbit_members.parquet")
                if (
                    o_path is None
                    or m_path is None
                    or not pathlib.Path(o_path).is_file()
                    or not pathlib.Path(m_path).is_file()
                ):
                    return None
                n_o = int(pq.ParquetFile(o_path).metadata.num_rows)
                n_m = int(pq.ParquetFile(m_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="initial_orbit_determination",
                    test_orbit_id=test_orbit_id,
                    path=(o_path, m_path),
                    stats={"num_iod_orbits": n_o, "num_iod_orbit_members": n_m},
                )
            if s == "differential_correction":
                o_path = _p("od_orbits.parquet")
                m_path = _p("od_orbit_members.parquet")
                if (
                    o_path is None
                    or m_path is None
                    or not pathlib.Path(o_path).is_file()
                    or not pathlib.Path(m_path).is_file()
                ):
                    return None
                n_o = int(pq.ParquetFile(o_path).metadata.num_rows)
                n_m = int(pq.ParquetFile(m_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="differential_correction",
                    test_orbit_id=test_orbit_id,
                    path=(o_path, m_path),
                    stats={"num_od_orbits": n_o, "num_od_orbit_members": n_m},
                )
            if s == "recover_orbits":
                ro_path = _p("recovered_orbits.parquet")
                rm_path = _p("recovered_orbit_members.parquet")
                if (
                    ro_path is None
                    or rm_path is None
                    or not pathlib.Path(ro_path).is_file()
                    or not pathlib.Path(rm_path).is_file()
                ):
                    return None
                n_o = int(pq.ParquetFile(ro_path).metadata.num_rows)
                n_m = int(pq.ParquetFile(rm_path).metadata.num_rows)
                return LinkTestOrbitStageResult(
                    name="recover_orbits",
                    test_orbit_id=test_orbit_id,
                    path=(ro_path, rm_path),
                    stats={"num_recovered_orbits": n_o, "num_recovered_orbit_members": n_m},
                )
        except Exception:
            # Best-effort only; if metadata read fails, fall back to running.
            return None
        return None

    stop_rank = _stage_rank(stop_after_stage)
    next_rank = _stage_rank(next_stage)
    if stop_rank is not None and next_rank is not None and next_rank > stop_rank:
        logger.info(
            "Checkpoint indicates next_stage=%s but stop_after_stage=%s; "
            "skipping execution beyond stop_after_stage",
            next_stage,
            stop_after_stage,
        )
        # If the stop stage output exists on disk, yield its result (so callers
        # can do bookkeeping / syncing) and exit. Otherwise, clamp to the stop
        # stage and let the normal stage runner recompute it.
        stop_stage_str = str(stop_after_stage).strip()
        ck = _checkpoint_stage_result(stop_stage_str)
        if ck is not None:
            yield ck
            return
        next_stage = stop_stage_str

    if next_stage == "complete":
        logger.info("Found recovered orbits on disk; exiting early...")
        ro_path = _p("recovered_orbits.parquet")
        rm_path = _p("recovered_orbit_members.parquet")
        assert ro_path is not None and rm_path is not None
        # We intentionally do not load the full tables here; the filesystem is
        # the source of truth and downstream consumers should read from paths.
        num_orbits = int(pq.ParquetFile(ro_path).metadata.num_rows)
        num_members = int(pq.ParquetFile(rm_path).metadata.num_rows)
        yield LinkTestOrbitStageResult(
            name="recover_orbits",
            test_orbit_id=test_orbit_id,
            path=(ro_path, rm_path),
            stats={
                "num_recovered_orbits": num_orbits,
                "num_recovered_orbit_members": num_members,
            },
        )
        return

    if next_stage == "filter_observations":
        filters = []
        if config.filter_cell_radius is not None:
            filters.append(TestOrbitRadiusObservationFilter(radius=config.filter_cell_radius))
        if config.filter_mahalanobis_distance is not None:
            filters.append(
                TestOrbitMahalanobisObservationFilter(mahalanobis_distance=config.filter_mahalanobis_distance)
            )

        # filter_observations now returns both filtered observations and captured ephemeris
        filtered_observations, test_orbit_ephemeris = filter_observations(
            observations,
            test_orbit,
            filters,
            propagator_class=propagator_class,
            max_processes=config.max_processes,
            chunk_size=config.filter_chunk_size,
        )

        filtered_observations_path = None
        n_filtered_observations = int(len(filtered_observations))
        if test_orbit_directory is not None:
            filtered_observations_path = os.path.join(test_orbit_directory, "filtered_observations.parquet")
            filtered_observations.to_parquet(filtered_observations_path)
            _log_parquet_metadata(filtered_observations_path, "after filter_observations stage")
            # IMPORTANT: When running with a working directory (the common pipeline
            # case), downstream stages reload from disk. Drop large in-memory
            # intermediates *before* yielding so they are not retained if the
            # caller stops iteration early (e.g. recursive splitting).
            filtered_observations = None  # type: ignore[assignment]
            test_orbit_ephemeris = None  # type: ignore[assignment]
            observations = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="filter_observations",
            test_orbit_id=test_orbit_id,
            path=(filtered_observations_path,),
            stats={"num_filtered_observations": n_filtered_observations},
        )
        if stop_after_stage == "filter_observations":
            return
        # Best-effort: drop large intermediates (safe no-ops when already None).
        filtered_observations = None  # type: ignore[assignment]
        test_orbit_ephemeris = None  # type: ignore[assignment]
        next_stage = "generate_ephemeris"

    # Observations are no longer needed
    del observations

    if next_stage == "generate_ephemeris":
        fo_path = _p("filtered_observations.parquet")
        assert fo_path is not None
        filtered_observations = Observations.from_parquet(fo_path)
        test_orbit_ephemeris_path = _p("test_orbit_ephemeris.parquet")
        test_orbit_ephemeris = (
            TestOrbitEphemeris.from_parquet(test_orbit_ephemeris_path)
            if test_orbit_ephemeris_path is not None and os.path.exists(test_orbit_ephemeris_path)
            else TestOrbitEphemeris.empty()
        )

        # If ephemeris from filtering is empty, generate it from filtered observations
        if len(test_orbit_ephemeris) == 0:
            logger.info("No ephemeris from filtering, generating from filtered observations...")
            test_orbit_ephemeris = test_orbit.generate_ephemeris_from_observations(
                filtered_observations,
                propagator_class=propagator_class,
                max_processes=config.max_processes,
                covariance=True,
            )

        ephemeris_path = None
        n_ephemeris_rows = int(len(test_orbit_ephemeris))
        if test_orbit_directory is not None:
            ephemeris_path = os.path.join(test_orbit_directory, "test_orbit_ephemeris.parquet")
            test_orbit_ephemeris.to_parquet(ephemeris_path)
            # In file-based mode, downstream stages reload from disk. Drop large
            # intermediates BEFORE yielding so callers don't retain them while
            # they do per-stage bookkeeping (e.g. rsync).
            filtered_observations = None  # type: ignore[assignment]
            test_orbit_ephemeris = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="generate_ephemeris",
            test_orbit_id=test_orbit_id,
            path=(ephemeris_path,),
            stats={"num_test_orbit_ephemeris_rows": n_ephemeris_rows},
        )
        if stop_after_stage == "generate_ephemeris":
            return
        next_stage = "range_and_transform"

    if next_stage == "range_and_transform":
        fo_path = _p("filtered_observations.parquet")
        eph_path = _p("test_orbit_ephemeris.parquet")
        assert fo_path is not None and eph_path is not None
        filtered_observations = Observations.from_parquet(fo_path)
        test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(eph_path)

        # Range and transform the observations
        transformed_detections = range_and_transform(
            test_orbit,
            filtered_observations,
            propagator_class=propagator_class,
            max_processes=config.max_processes,
            test_orbit_ephemeris=test_orbit_ephemeris,
        )

        transformed_detections_path = None
        n_transformed = int(len(transformed_detections))
        if test_orbit_directory is not None:
            logger.info(f"Saving transformed detections to {test_orbit_directory}...")
            transformed_detections_path = os.path.join(test_orbit_directory, "transformed_detections.parquet")
            transformed_detections.to_parquet(transformed_detections_path)
            # In file-based mode, downstream stages reload from disk. Drop large
            # intermediates BEFORE yielding so callers don't retain them while
            # they do per-stage bookkeeping (e.g. rsync).
            filtered_observations = None  # type: ignore[assignment]
            test_orbit_ephemeris = None  # type: ignore[assignment]
            transformed_detections = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="range_and_transform",
            test_orbit_id=test_orbit_id,
            path=(transformed_detections_path,),
            stats={"num_transformed_detections": n_transformed},
        )
        if stop_after_stage == "range_and_transform":
            return
        next_stage = "cluster_and_link"

    if next_stage == "cluster_and_link":
        td_path = _p("transformed_detections.parquet")
        eph_path = _p("test_orbit_ephemeris.parquet")
        assert td_path is not None and eph_path is not None
        transformed_detections = TransformedDetections.from_parquet(td_path)
        test_orbit_ephemeris = TestOrbitEphemeris.from_parquet(eph_path)

        # Run clustering (parameters calculated automatically from test_orbit_ephemeris if provided)
        clusters, cluster_members = cluster_and_link(
            transformed_detections,
            test_orbit_ephemeris=test_orbit_ephemeris,
            velocity_bin_separation=config.cluster_velocity_bin_separation,
            min_obs=config.cluster_min_obs,
            min_arc_length=config.cluster_min_arc_length,
            min_nights=config.cluster_min_nights,
            rchi2_threshold=config.cluster_rchi2_threshold,
            mahalanobis_distance=config.cluster_mahalanobis_distance,
            alg=config.cluster_algorithm,
            radius=config.cluster_radius,
            vx_range=[config.cluster_vx_min, config.cluster_vx_max],
            vy_range=[config.cluster_vy_min, config.cluster_vy_max],
            vx_bins=config.cluster_vx_bins,
            vy_bins=config.cluster_vy_bins,
            vx_values=None,
            vy_values=None,
            chunk_size=config.cluster_chunk_size,
            max_processes=config.max_processes,
            whiten=config.cluster_whiten,
        )

        clusters_path = None
        cluster_members_path = None
        n_clusters = int(len(clusters))
        n_cluster_members = int(len(cluster_members))
        if test_orbit_directory is not None:
            logger.info(f"Saving clusters to {test_orbit_directory}...")
            clusters_path = os.path.join(test_orbit_directory, "clusters.parquet")
            cluster_members_path = os.path.join(test_orbit_directory, "cluster_members.parquet")
            clusters.to_parquet(clusters_path)
            cluster_members.to_parquet(cluster_members_path)
            transformed_detections = None  # type: ignore[assignment]
            test_orbit_ephemeris = None  # type: ignore[assignment]
            clusters = None  # type: ignore[assignment]
            cluster_members = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="cluster_and_link",
            test_orbit_id=test_orbit_id,
            path=(clusters_path, cluster_members_path),
            stats={
                "num_clusters": n_clusters,
                "num_cluster_members": n_cluster_members,
            },
        )
        if stop_after_stage == "cluster_and_link":
            return
        next_stage = "initial_orbit_determination"

    if next_stage == "initial_orbit_determination":
        fo_path = _p("filtered_observations.parquet")
        c_path = _p("clusters.parquet")
        cm_path = _p("cluster_members.parquet")
        assert fo_path is not None and c_path is not None and cm_path is not None
        filtered_observations = Observations.from_parquet(fo_path)
        clusters = FittedClusters.from_parquet(c_path)
        cluster_members = FittedClusterMembers.from_parquet(cm_path)

        # Run initial orbit determination
        iod_orbits, iod_orbit_members = initial_orbit_determination(
            filtered_observations,
            cluster_members,
            propagator_class=propagator_class,
            min_obs=config.iod_min_obs,
            min_arc_length=config.iod_min_arc_length,
            contamination_percentage=config.iod_contamination_percentage,
            rchi2_threshold=config.iod_rchi2_threshold,
            observation_selection_method=config.iod_observation_selection_method,
            iterate=False,
            light_time=True,
            linkage_id_col="cluster_id",
            propagator_kwargs={},
            chunk_size=config.iod_chunk_size,
            max_processes=config.max_processes,
        )

        iod_orbits_path = None
        iod_orbit_members_path = None
        n_iod_orbits = int(len(iod_orbits))
        n_iod_orbit_members = int(len(iod_orbit_members))
        if test_orbit_directory is not None:
            logger.info(f"Saving IOD orbits to {test_orbit_directory}...")
            iod_orbits_path = os.path.join(test_orbit_directory, "iod_orbits.parquet")
            iod_orbit_members_path = os.path.join(test_orbit_directory, "iod_orbit_members.parquet")
            iod_orbits.to_parquet(iod_orbits_path)
            iod_orbit_members.to_parquet(iod_orbit_members_path)
            filtered_observations = None  # type: ignore[assignment]
            clusters = None  # type: ignore[assignment]
            cluster_members = None  # type: ignore[assignment]
            iod_orbits = None  # type: ignore[assignment]
            iod_orbit_members = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="initial_orbit_determination",
            test_orbit_id=test_orbit_id,
            path=(iod_orbits_path, iod_orbit_members_path),
            stats={
                "num_iod_orbits": n_iod_orbits,
                "num_iod_orbit_members": n_iod_orbit_members,
            },
        )
        if stop_after_stage == "initial_orbit_determination":
            return
        next_stage = "differential_correction"

    if next_stage == "differential_correction":
        fo_path = _p("filtered_observations.parquet")
        iod_o_path = _p("iod_orbits.parquet")
        iod_m_path = _p("iod_orbit_members.parquet")
        assert fo_path is not None and iod_o_path is not None and iod_m_path is not None
        filtered_observations = Observations.from_parquet(fo_path)
        iod_orbits = FittedOrbits.from_parquet(iod_o_path)
        iod_orbit_members = FittedOrbitMembers.from_parquet(iod_m_path)

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
            propagator_class=propagator_class,
            propagator_kwargs={},
            chunk_size=config.od_chunk_size,
            max_processes=config.max_processes,
            # TODO: investigate whether these should be configurable
            method="central",
        )

        od_orbits_path = None
        od_orbit_members_path = None
        n_od_orbits = int(len(od_orbits))
        n_od_orbit_members = int(len(od_orbit_members))
        if test_orbit_directory is not None:
            logger.info(f"Saving OD orbits to {test_orbit_directory}...")
            od_orbits_path = os.path.join(test_orbit_directory, "od_orbits.parquet")
            od_orbit_members_path = os.path.join(test_orbit_directory, "od_orbit_members.parquet")
            od_orbits.to_parquet(od_orbits_path)
            od_orbit_members.to_parquet(od_orbit_members_path)
            filtered_observations = None  # type: ignore[assignment]
            iod_orbits = None  # type: ignore[assignment]
            iod_orbit_members = None  # type: ignore[assignment]
            od_orbits = None  # type: ignore[assignment]
            od_orbit_members = None  # type: ignore[assignment]

        yield LinkTestOrbitStageResult(
            name="differential_correction",
            test_orbit_id=test_orbit_id,
            path=(od_orbits_path, od_orbit_members_path),
            stats={
                "num_od_orbits": n_od_orbits,
                "num_od_orbit_members": n_od_orbit_members,
            },
        )
        if stop_after_stage == "differential_correction":
            return
        next_stage = "recover_orbits"

    if next_stage == "recover_orbits":
        fo_path = _p("filtered_observations.parquet")
        od_o_path = _p("od_orbits.parquet")
        od_m_path = _p("od_orbit_members.parquet")
        assert fo_path is not None and od_o_path is not None and od_m_path is not None
        filtered_observations = Observations.from_parquet(fo_path)
        od_orbits = FittedOrbits.from_parquet(od_o_path)
        od_orbit_members = FittedOrbitMembers.from_parquet(od_m_path)

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
            propagator_class=propagator_class,
            propagator_kwargs={},
            orbits_chunk_size=config.arc_extension_chunk_size,
            max_processes=config.max_processes,
            # TODO: investigate whether these should be configurable
            method="central",
            observations_chunk_size=100000,
        )

        recovered_orbits_path = None
        recovered_orbit_members_path = None
        n_recovered_orbits = int(len(recovered_orbits))
        n_recovered_members = int(len(recovered_orbit_members))
        if test_orbit_directory is not None:
            logger.info(f"Saving recovered orbits to {test_orbit_directory}...")
            recovered_orbits_path = os.path.join(test_orbit_directory, "recovered_orbits.parquet")
            recovered_orbit_members_path = os.path.join(
                test_orbit_directory, "recovered_orbit_members.parquet"
            )
            recovered_orbits.to_parquet(recovered_orbits_path)
            recovered_orbit_members.to_parquet(recovered_orbit_members_path)
            filtered_observations = None  # type: ignore[assignment]
            od_orbits = None  # type: ignore[assignment]
            od_orbit_members = None  # type: ignore[assignment]
            recovered_orbits = None  # type: ignore[assignment]
            recovered_orbit_members = None  # type: ignore[assignment]

        time_end = time.perf_counter()
        logger.info(f"Test orbit completed in {time_end-time_start:.3f} seconds.")
        yield LinkTestOrbitStageResult(
            name="recover_orbits",
            test_orbit_id=test_orbit_id,
            path=(recovered_orbits_path, recovered_orbit_members_path),
            stats={
                "num_recovered_orbits": n_recovered_orbits,
                "num_recovered_orbit_members": n_recovered_members,
            },
        )


@dataclass
class _OrbitWorkItem:
    """
    Work item for the iterative recursive-splitting trampoline in `link_test_orbits`.

    This replaces the previous `tuple[...]` stack entries with a named, typed object.
    """

    test_orbit: TestOrbits
    observations: Union[str, Observations]
    working_dir: Optional[str]
    use_orbit_subdir: bool
    split_depth: int


def link_test_orbits(
    test_orbits: TestOrbits,
    observations: Union[str, Observations],
    working_dir: Optional[str] = None,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
    use_orbit_subdir: bool = True,
    max_split_depth: int = 2,
    current_depth: int = 0,
) -> Iterator[LinkTestOrbitStageResult]:
    """
    Run THOR for a list of test orbits on the given observations. This function will yield
    results at each stage of the pipeline.

    Parameters
    ----------
    test_orbits : `~thor.orbit.TestOrbits`
        Collection of test orbits to process.
    observations : `~thor.observations.observations.Observations` or str
        Observations to search for moving objects.
    working_dir : str, optional
        Directory with persisted config and checkpointed results.
    filters : list of `~thor.observations.filters.ObservationFilter`, optional
        List of filters to apply to the observations before running THOR.
    config : `~thor.config.Config`, optional
        Configuration to use for THOR. If None, the default configuration will be used.
    current_depth: int, optional
        Current depth of the split.

    Returns
    -------
    Iterator[LinkTestOrbitStageResult]
        Iterator over the results of each stage of the pipeline for each test orbit.
    """
    if config is None:
        config = Config()

    split_threshold = config.split_threshold
    split_method = config.split_method
    max_split_depth = config.split_max_depth

    # ----------------------------------------------------------------------
    # Iterative "trampoline" recursion using an explicit stack.
    #
    # This avoids `yield from link_test_orbits(...)` (which is elegant but can
    # make lifecycle/memory interactions harder to reason about) and makes it
    # straightforward to ensure per-orbit generators are always closed when we
    # stop early due to splitting.
    # ----------------------------------------------------------------------
    orbit_work_stack: List[_OrbitWorkItem] = []
    # Preserve the original processing order by pushing in reverse and popping
    # from the end (LIFO depth-first, matching the prior recursive behavior).
    test_orbit_rows_in_order = [test_orbit_row for test_orbit_row in test_orbits]
    for test_orbit_row in reversed(test_orbit_rows_in_order):
        orbit_work_stack.append(
            _OrbitWorkItem(
                test_orbit=test_orbit_row,
                observations=observations,
                working_dir=working_dir,
                use_orbit_subdir=use_orbit_subdir,
                split_depth=current_depth,
            )
        )

    while orbit_work_stack:
        work_item = orbit_work_stack.pop()
        test_orbit = work_item.test_orbit
        observations_input = work_item.observations
        working_directory = work_item.working_dir
        use_orbit_subdirectory = work_item.use_orbit_subdir
        split_depth = work_item.split_depth

        # ------------------------------------------------------------------
        # Recursive split resume support (from on-disk split plan)
        # ------------------------------------------------------------------
        if working_directory is not None:
            orbit_id_str = test_orbit.orbit_id[0].as_py()
            parent_orbit_directory = (
                pathlib.Path(working_directory, orbit_id_str)
                if use_orbit_subdirectory
                else pathlib.Path(working_directory)
            )
            split_plan_path = parent_orbit_directory / "split_test_orbits.parquet"
            if split_plan_path.exists():
                try:
                    split_test_orbits = TestOrbits.from_parquet(split_plan_path)
                    if split_test_orbits.fragmented():
                        split_test_orbits = qv.defragment(split_test_orbits)
                except Exception:
                    split_test_orbits = None  # type: ignore[assignment]

                if split_test_orbits is not None:
                    filtered_observations_path = (
                        parent_orbit_directory / "filtered_observations.parquet"
                    )
                    if filtered_observations_path.exists():
                        child_observations = str(filtered_observations_path)
                        _log_parquet_metadata(
                            str(filtered_observations_path),
                            "reading filtered_observations for split orbit resume",
                        )
                    else:
                        child_observations = observations_input

                    # Push children so they run before any remaining siblings on
                    # the stack (depth-first).
                    child_test_orbit_rows = [
                        child_test_orbit_row for child_test_orbit_row in split_test_orbits
                    ]
                    for child_test_orbit_row in reversed(child_test_orbit_rows):
                        orbit_work_stack.append(
                            _OrbitWorkItem(
                                test_orbit=child_test_orbit_row,
                                observations=child_observations,
                                working_dir=str(parent_orbit_directory),
                                use_orbit_subdir=True,
                                split_depth=split_depth + 1,
                            )
                        )
                    continue

        # Normal pipeline execution for this orbit
        stage_iter = link_test_orbit(
            test_orbit,
            observations_input,
            working_directory,
            filters,
            config,
            use_orbit_subdir=use_orbit_subdirectory,
        )
        split_triggered = False
        try:
            for stage_result in stage_iter:
                if (
                    split_threshold is not None
                    and stage_result.name == "filter_observations"
                    and int(stage_result.stats.get("num_filtered_observations", 0)) > split_threshold
                ):
                    if split_depth >= max_split_depth:
                        logger.info(
                            f"Split threshold exceeded but max depth {max_split_depth} reached; not splitting further.",
                        )
                        # Do not split further, but continue running downstream stages for this orbit.
                        yield stage_result
                        continue

                    logger.info(
                        "Filtered observations (%s) exceed threshold (%s); splitting test orbit via %s.",
                        int(stage_result.stats.get("num_filtered_observations", 0)),
                        split_threshold,
                        split_method,
                    )
                    # Use the on-disk path if available; otherwise fall back to the
                    # original observations input (in-memory mode only).
                    if stage_result.path and stage_result.path[0] is not None:
                        child_observations = stage_result.path[0]
                    else:
                        child_observations = observations_input

                    if split_method == "healpixel":
                        split_test_orbits = test_orbit.split_healpixel()
                    else:
                        observation_times = get_observation_times(child_observations)
                        max_time = observation_times.max()
                        min_time = observation_times.min()
                        time_span_days = max_time.mjd()[0].as_py() - min_time.mjd()[0].as_py()
                        try:
                            del observation_times
                        except Exception:
                            pass
                        split_test_orbits = test_orbit.split(
                            dt=time_span_days, k=1, beta=0.67, gamma=1, num=3, max_depth=1
                        )

                    parent_orbit_directory: Optional[pathlib.Path] = None
                    if working_directory is not None:
                        parent_orbit_directory = (
                            pathlib.Path(working_directory, test_orbit.orbit_id[0].as_py())
                            if use_orbit_subdirectory
                            else pathlib.Path(working_directory)
                        )
                        parent_orbit_directory.mkdir(parents=True, exist_ok=True)
                        try:
                            split_plan_path = parent_orbit_directory / "split_test_orbits.parquet"
                            tmp_path = parent_orbit_directory / "split_test_orbits.parquet.tmp"
                            split_test_orbits.to_parquet(str(tmp_path))
                            tmp_path.replace(split_plan_path)
                        except Exception:
                            logger.exception(
                                "Failed to persist split_test_orbits under %s; recursion may not be resumable",
                                parent_orbit_directory,
                            )

                    # Encourage prompt cleanup before launching children.
                    try:
                        import gc

                        gc.collect()
                    except Exception:
                        pass

                    # Push children so they run before any remaining siblings on
                    # the stack (depth-first).
                    child_working_directory = (
                        str(parent_orbit_directory) if parent_orbit_directory is not None else None
                    )
                    child_test_orbit_rows = [
                        child_test_orbit_row for child_test_orbit_row in split_test_orbits
                    ]
                    for child_test_orbit_row in reversed(child_test_orbit_rows):
                        orbit_work_stack.append(
                            _OrbitWorkItem(
                                test_orbit=child_test_orbit_row,
                                observations=child_observations,
                                working_dir=child_working_directory,
                                use_orbit_subdir=True,
                                split_depth=split_depth + 1,
                            )
                        )

                    split_triggered = True
                    break

                yield stage_result
        finally:
            # Always close the per-orbit generator to ensure its frame doesn't
            # retain large locals after early exit.
            try:
                stage_iter.close()
            except Exception:
                pass

        if split_triggered:
            continue
