import importlib
import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Tuple, Union

import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray

from .checkpointing import create_checkpoint_data, load_initial_checkpoint_values
from .clustering import (
    HoughLineClustering,
    VelocityGridDBSCAN,
    VelocityGridFFT,
    VelocityGridHotspot2D,
    VelocityGridKDTree,
    VelocityGridOPTICS,
    fit_clusters,
    form_tracklets,
)

try:
    from .clustering import CUDAShiftAndStack

    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

from .config import Config, initialize_config
from .observations.filters import (
    ObservationFilter,
    TestOrbitMahalanobisObservationFilter,
    TestOrbitRadiusObservationFilter,
    filter_observations,
)
from .observations.observations import Observations, get_observation_times
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
        "generate_ephemeris",
        "range_and_transform",
        "form_tracklets",
        "cluster_and_link",
        "fit_clusters",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
    ]
    result: Tuple[Union[qv.AnyTable, str], ...]
    path: Tuple[Optional[str], ...] = (None,)


def link_test_orbit(
    test_orbit: TestOrbits,
    observations: Union[str, Observations],
    working_dir: Optional[str] = None,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
    use_orbit_subdir: bool = True,
    yield_paths: bool = False,
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
    yield_paths : bool, optional
        If True and working_dir is set, yield file paths in result instead of in-memory
        tables. This allows large tables to be garbage collected after being written to
        disk, reducing memory usage. Default is False.
    """
    time_start = time.perf_counter()
    logger.info("Running test orbit...")

    if len(test_orbit) != 1:
        raise ValueError(f"link_test_orbit received {len(test_orbit)} orbits but expected 1.")

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

    use_ray = initialize_use_ray(
        num_cpus=config.max_processes,
        object_store_bytes=config.ray_memory_bytes or None,
    )

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
        if test_orbit_directory is not None:
            filtered_observations_path = os.path.join(test_orbit_directory, "filtered_observations.parquet")
            filtered_observations.to_parquet(filtered_observations_path)

        yield LinkTestOrbitStageResult(
            name="filter_observations",
            result=(
                (
                    filtered_observations_path
                    if yield_paths and filtered_observations_path
                    else filtered_observations
                ),
            ),
            path=(filtered_observations_path,),
        )
        if stop_after_stage == "filter_observations":
            return

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")

        checkpoint = create_checkpoint_data(
            "generate_ephemeris",
            filtered_observations=filtered_observations,
            test_orbit_ephemeris=test_orbit_ephemeris,
        )

    # Observations are no longer needed
    del observations

    if checkpoint.stage == "generate_ephemeris":
        filtered_observations = checkpoint.filtered_observations
        test_orbit_ephemeris = checkpoint.test_orbit_ephemeris

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
        if test_orbit_directory is not None:
            ephemeris_path = os.path.join(test_orbit_directory, "test_orbit_ephemeris.parquet")
            test_orbit_ephemeris.to_parquet(ephemeris_path)

        yield LinkTestOrbitStageResult(
            name="generate_ephemeris",
            result=(ephemeris_path if yield_paths and ephemeris_path else test_orbit_ephemeris,),
            path=(ephemeris_path,),
        )
        if stop_after_stage == "generate_ephemeris":
            return

        if use_ray:
            if not isinstance(test_orbit_ephemeris, ray.ObjectRef):
                test_orbit_ephemeris = ray.put(test_orbit_ephemeris)
                logger.info("Placed test orbit ephemeris in the object store.")
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")

        checkpoint = create_checkpoint_data(
            "range_and_transform",
            test_orbit_ephemeris=test_orbit_ephemeris,
            filtered_observations=filtered_observations,
        )

    if checkpoint.stage == "range_and_transform":
        test_orbit_ephemeris = checkpoint.test_orbit_ephemeris
        filtered_observations = checkpoint.filtered_observations
        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")

        # Range and transform the observations
        transformed_detections = range_and_transform(
            test_orbit,
            filtered_observations,
            propagator_class=propagator_class,
            max_processes=config.max_processes,
            test_orbit_ephemeris=test_orbit_ephemeris,
        )

        transformed_detections_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving transformed detections to {test_orbit_directory}...")
            transformed_detections_path = os.path.join(test_orbit_directory, "transformed_detections.parquet")
            transformed_detections.to_parquet(transformed_detections_path)

        yield LinkTestOrbitStageResult(
            name="range_and_transform",
            result=(
                (
                    transformed_detections_path
                    if yield_paths and transformed_detections_path
                    else transformed_detections
                ),
            ),
            path=(transformed_detections_path,),
        )
        if stop_after_stage == "range_and_transform":
            return

        if use_ray:
            if not isinstance(test_orbit_ephemeris, ray.ObjectRef):
                test_orbit_ephemeris = ray.put(test_orbit_ephemeris)
                logger.info("Placed test orbit ephemeris in the object store.")
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(transformed_detections, ray.ObjectRef):
                transformed_detections = ray.put(transformed_detections)
                logger.info("Placed transformed detections in the object store.")

        checkpoint = create_checkpoint_data(
            "form_tracklets",
            test_orbit_ephemeris=test_orbit_ephemeris,
            filtered_observations=filtered_observations,
            transformed_detections=transformed_detections,
        )

    if checkpoint.stage == "form_tracklets":
        test_orbit_ephemeris = checkpoint.test_orbit_ephemeris
        filtered_observations = checkpoint.filtered_observations
        transformed_detections = checkpoint.transformed_detections

        tracklets = None
        tracklet_members = None
        if config.use_tracklets:
            if isinstance(transformed_detections, ray.ObjectRef):
                td_local = ray.get(transformed_detections)
            else:
                td_local = transformed_detections
            if isinstance(test_orbit_ephemeris, ray.ObjectRef):
                toe_local = ray.get(test_orbit_ephemeris)
            else:
                toe_local = test_orbit_ephemeris

            tracklets, tracklet_members = form_tracklets(
                td_local,
                toe_local,
                min_obs=config.tracklet_min_obs,
                max_velocity=config.tracklet_max_velocity,
                mahalanobis_distance=config.tracklet_mahalanobis_distance,
            )

        tracklets_path = None
        tracklet_members_path = None
        if test_orbit_directory is not None and tracklets is not None:
            logger.info(f"Saving tracklets to {test_orbit_directory}...")
            tracklets_path = os.path.join(test_orbit_directory, "tracklets.parquet")
            tracklet_members_path = os.path.join(test_orbit_directory, "tracklet_members.parquet")
            tracklets.to_parquet(tracklets_path)
            tracklet_members.to_parquet(tracklet_members_path)

        yield LinkTestOrbitStageResult(
            name="form_tracklets",
            result=(
                tracklets_path if yield_paths and tracklets_path else tracklets,
                tracklet_members_path if yield_paths and tracklet_members_path else tracklet_members,
            ),
            path=(tracklets_path, tracklet_members_path),
        )
        if stop_after_stage == "form_tracklets":
            return

        if use_ray:
            if not isinstance(test_orbit_ephemeris, ray.ObjectRef):
                test_orbit_ephemeris = ray.put(test_orbit_ephemeris)
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
            if not isinstance(transformed_detections, ray.ObjectRef):
                transformed_detections = ray.put(transformed_detections)

        checkpoint = create_checkpoint_data(
            "cluster_and_link",
            test_orbit_ephemeris=test_orbit_ephemeris,
            filtered_observations=filtered_observations,
            transformed_detections=transformed_detections,
            tracklets=tracklets,
            tracklet_members=tracklet_members,
        )

    if checkpoint.stage == "cluster_and_link":
        test_orbit_ephemeris = checkpoint.test_orbit_ephemeris
        filtered_observations = checkpoint.filtered_observations
        transformed_detections = checkpoint.transformed_detections
        tracklets = getattr(checkpoint, "tracklets", None)
        tracklet_members = getattr(checkpoint, "tracklet_members", None)

        # Instantiate the clustering algorithm from config
        _algorithm_classes = {
            "dbscan": VelocityGridDBSCAN,
            "hotspot_2d": VelocityGridHotspot2D,
            "optics": VelocityGridOPTICS,
            "kdtree": VelocityGridKDTree,
            "fft": VelocityGridFFT,
            "hough": HoughLineClustering,
        }
        if _CUDA_AVAILABLE:
            _algorithm_classes["cuda"] = CUDAShiftAndStack
        if config.cluster_algorithm not in _algorithm_classes:
            raise NotImplementedError(f"algorithm '{config.cluster_algorithm}' is not implemented")

        clustering_algorithm = _algorithm_classes[config.cluster_algorithm](
            radius=config.cluster_radius,
            min_obs=config.cluster_min_obs,
            min_arc_length=config.cluster_min_arc_length,
            min_nights=config.cluster_min_nights,
            vx_range=[config.cluster_vx_min, config.cluster_vx_max],
            vy_range=[config.cluster_vy_min, config.cluster_vy_max],
            vx_bins=config.cluster_vx_bins,
            vy_bins=config.cluster_vy_bins,
            velocity_bin_separation=config.cluster_velocity_bin_separation,
            mahalanobis_distance=config.cluster_mahalanobis_distance,
            radius_multiplier=config.cluster_radius_multiplier,
            density_multiplier=config.cluster_density_multiplier,
            min_radius=config.cluster_min_radius,
            max_radius=config.cluster_max_radius,
            chunk_size=config.cluster_chunk_size,
            max_processes=config.max_processes,
            whiten=config.cluster_whiten,
            astrometric_precision=config.cluster_astrometric_precision,
            window_enabled=config.cluster_window_enabled,
            window_min_days=config.cluster_window_min_days,
        )

        # Run clustering (finding only — no fitting)
        clusters, cluster_members = clustering_algorithm.find_clusters(
            transformed_detections,
            test_orbit_ephemeris=test_orbit_ephemeris,
            tracklets=tracklets,
            tracklet_members=tracklet_members,
        )

        clusters_path = None
        cluster_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving unfitted clusters to {test_orbit_directory}...")
            clusters_path = os.path.join(test_orbit_directory, "clusters.parquet")
            cluster_members_path = os.path.join(test_orbit_directory, "cluster_members.parquet")
            clusters.to_parquet(clusters_path)
            cluster_members.to_parquet(cluster_members_path)

        yield LinkTestOrbitStageResult(
            name="cluster_and_link",
            result=(
                clusters_path if yield_paths and clusters_path else clusters,
                cluster_members_path if yield_paths and cluster_members_path else cluster_members,
            ),
            path=(clusters_path, cluster_members_path),
        )
        if stop_after_stage == "cluster_and_link":
            return

        checkpoint = create_checkpoint_data(
            "fit_clusters",
            filtered_observations=filtered_observations,
            clusters=clusters,
            cluster_members=cluster_members,
            transformed_detections=transformed_detections,
        )

    if checkpoint.stage == "fit_clusters":
        filtered_observations = checkpoint.filtered_observations
        clusters = checkpoint.clusters
        cluster_members = checkpoint.cluster_members
        transformed_detections = checkpoint.transformed_detections

        # Fit clusters with polynomial motion model
        fitted_clusters, fitted_cluster_members = fit_clusters(
            clusters,
            cluster_members,
            transformed_detections,
            rchi2_threshold=config.cluster_rchi2_threshold,
            chunk_size=config.cluster_chunk_size,
            max_processes=config.max_processes,
        )

        # transformed_detections no longer needed after fitting
        del transformed_detections

        fitted_clusters_path = None
        fitted_cluster_members_path = None
        if test_orbit_directory is not None:
            logger.info(f"Saving fitted clusters to {test_orbit_directory}...")
            fitted_clusters_path = os.path.join(test_orbit_directory, "fitted_clusters.parquet")
            fitted_cluster_members_path = os.path.join(test_orbit_directory, "fitted_cluster_members.parquet")
            fitted_clusters.to_parquet(fitted_clusters_path)
            fitted_cluster_members.to_parquet(fitted_cluster_members_path)

        yield LinkTestOrbitStageResult(
            name="fit_clusters",
            result=(
                fitted_clusters_path if yield_paths and fitted_clusters_path else fitted_clusters,
                (
                    fitted_cluster_members_path
                    if yield_paths and fitted_cluster_members_path
                    else fitted_cluster_members
                ),
            ),
            path=(fitted_clusters_path, fitted_cluster_members_path),
        )
        if stop_after_stage == "fit_clusters":
            return

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(fitted_clusters, ray.ObjectRef):
                fitted_clusters = ray.put(fitted_clusters)
                logger.info("Placed fitted clusters in the object store.")
            if not isinstance(fitted_cluster_members, ray.ObjectRef):
                fitted_cluster_members = ray.put(fitted_cluster_members)
                logger.info("Placed fitted cluster members in the object store.")

        checkpoint = create_checkpoint_data(
            "initial_orbit_determination",
            filtered_observations=filtered_observations,
            clusters=fitted_clusters,
            cluster_members=fitted_cluster_members,
        )

    if checkpoint.stage == "initial_orbit_determination":
        filtered_observations = checkpoint.filtered_observations
        clusters = checkpoint.clusters
        cluster_members = checkpoint.cluster_members

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
        if test_orbit_directory is not None:
            logger.info(f"Saving IOD orbits to {test_orbit_directory}...")
            iod_orbits_path = os.path.join(test_orbit_directory, "iod_orbits.parquet")
            iod_orbit_members_path = os.path.join(test_orbit_directory, "iod_orbit_members.parquet")
            iod_orbits.to_parquet(iod_orbits_path)
            iod_orbit_members.to_parquet(iod_orbit_members_path)

        yield LinkTestOrbitStageResult(
            name="initial_orbit_determination",
            result=(
                iod_orbits_path if yield_paths and iod_orbits_path else iod_orbits,
                iod_orbit_members_path if yield_paths and iod_orbit_members_path else iod_orbit_members,
            ),
            path=(iod_orbits_path, iod_orbit_members_path),
        )
        if stop_after_stage == "initial_orbit_determination":
            return

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(iod_orbits, ray.ObjectRef):
                iod_orbits = ray.put(iod_orbits)
                logger.info("Placed initial orbits in the object store.")
            if not isinstance(iod_orbit_members, ray.ObjectRef):
                iod_orbit_members = ray.put(iod_orbit_members)
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
            propagator_class=propagator_class,
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
            result=(
                od_orbits_path if yield_paths and od_orbits_path else od_orbits,
                od_orbit_members_path if yield_paths and od_orbit_members_path else od_orbit_members,
            ),
            path=(od_orbits_path, od_orbit_members_path),
        )
        if stop_after_stage == "differential_correction":
            return

        if use_ray:
            if not isinstance(filtered_observations, ray.ObjectRef):
                filtered_observations = ray.put(filtered_observations)
                logger.info("Placed filtered observations in the object store.")
            if not isinstance(od_orbits, ray.ObjectRef):
                od_orbits = ray.put(od_orbits)
                logger.info("Placed differentially corrected orbits in the object store.")
            if not isinstance(od_orbit_members, ray.ObjectRef):
                od_orbit_members = ray.put(od_orbit_members)
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
            result=(
                recovered_orbits_path if yield_paths and recovered_orbits_path else recovered_orbits,
                (
                    recovered_orbit_members_path
                    if yield_paths and recovered_orbit_members_path
                    else recovered_orbit_members
                ),
            ),
            path=(recovered_orbits_path, recovered_orbit_members_path),
        )


def link_test_orbits(
    test_orbits: TestOrbits,
    observations: Union[str, Observations],
    working_dir: Optional[str] = None,
    filters: Optional[List[ObservationFilter]] = None,
    config: Optional[Config] = None,
    use_orbit_subdir: bool = True,
    current_depth: int = 0,
    yield_paths: bool = False,
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
    yield_paths : bool, optional
        If True and working_dir is set, yield file paths in result instead of in-memory
        tables. This allows large tables to be garbage collected after being written to
        disk, reducing memory usage. Default is False.

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

    for test_orbit in test_orbits:
        for stage_result in link_test_orbit(
            test_orbit,
            observations,
            working_dir,
            filters,
            config,
            use_orbit_subdir=use_orbit_subdir,
            yield_paths=yield_paths,
        ):
            if split_threshold is not None and stage_result.name == "filter_observations":
                # Get the number of filtered observations
                result_item = stage_result.result[0]
                if isinstance(result_item, str):
                    # Result is a path, read metadata to get row count
                    num_filtered = pq.read_metadata(result_item).num_rows
                else:
                    num_filtered = len(result_item)

                if num_filtered > split_threshold:
                    if current_depth >= max_split_depth:
                        logger.info(
                            f"Split threshold exceeded but max depth {max_split_depth} reached; not splitting further.",
                        )
                        # Do not split further, but continue running downstream stages for this orbit.
                        yield stage_result
                        continue
                    logger.info(
                        f"Filtered observations ({num_filtered}) exceed threshold ({split_threshold}); splitting test orbit via {split_method}.",
                    )

                    # Use the on-disk path if available, otherwise fall back to in-memory observations
                    child_observations = (
                        stage_result.path[0] if stage_result.path[0] is not None else stage_result.result[0]
                    )

                    if split_method == "healpixel":
                        split_test_orbits = test_orbit.split_healpixel()
                    else:
                        times = get_observation_times(child_observations)
                        max_time = times.max()
                        min_time = times.min()
                        dt = max_time.mjd()[0].as_py() - min_time.mjd()[0].as_py()
                        split_test_orbits = test_orbit.split(
                            dt=dt, k=1, beta=0.67, gamma=1, num=3, max_depth=1
                        )

                    if working_dir is not None:
                        parent_dir = (
                            pathlib.Path(working_dir, test_orbit.orbit_id[0].as_py())
                            if use_orbit_subdir
                            else pathlib.Path(working_dir)
                        )
                        parent_dir.mkdir(parents=True, exist_ok=True)

                    child_working_dir: Optional[str] = None
                    if working_dir is not None:
                        # Each child gets its own subdir inside the parent orbit directory.
                        child_working_dir = str(parent_dir)

                    yield from link_test_orbits(
                        split_test_orbits,
                        child_observations,
                        child_working_dir,
                        filters,
                        config,
                        use_orbit_subdir=True,
                        current_depth=current_depth + 1,
                        yield_paths=yield_paths,
                    )
                    break

            yield stage_result
