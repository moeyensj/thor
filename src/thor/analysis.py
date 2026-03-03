import logging
import pathlib
from typing import Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from difi import analyze_observations
from difi.difi import LinkageMembers, PartitionMapping, analyze_linkages
from difi.metrics import FindabilityMetric, SingletonMetric
from difi.observations import Observations as AnalysisObservations

from .clusters import FittedClusterMembers, FittedClusters
from .config import Config
from .observations.observations import Observations
from .orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits

__all__ = ["ObservationLabels", "analyze_orbit", "analyze_run"]

logger = logging.getLogger(__name__)


class ObservationLabels(qv.Table):
    obs_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)


class PipelineAllObjects(qv.Table):
    test_orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()
    mjd_min = qv.Float64Column()
    mjd_max = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    num_observations = qv.Int64Column()
    findable = qv.BooleanColumn()

    # Cluster and link
    found_pure_clusters = qv.Int64Column(default=0)
    found_contaminated_clusters = qv.Int64Column(default=0)
    pure_clusters = qv.Int64Column(default=0)
    pure_complete_clusters = qv.Int64Column(default=0)
    contaminated_clusters = qv.Int64Column(default=0)
    contaminant_clusters = qv.Int64Column(default=0)
    mixed_clusters = qv.Int64Column(default=0)
    obs_in_pure_clusters = qv.Int64Column(default=0)
    obs_in_pure_complete_clusters = qv.Int64Column(default=0)
    obs_in_contaminated_clusters = qv.Int64Column(default=0)
    obs_as_contaminant_clusters = qv.Int64Column(default=0)
    obs_in_mixed_clusters = qv.Int64Column(default=0)

    # Initial orbit determination
    found_pure_iod = qv.Int64Column(default=0)
    found_contaminated_iod = qv.Int64Column(default=0)
    pure_iod = qv.Int64Column(default=0)
    pure_complete_iod = qv.Int64Column(default=0)
    contaminated_iod = qv.Int64Column(default=0)
    contaminant_iod = qv.Int64Column(default=0)
    mixed_iod = qv.Int64Column(default=0)
    obs_in_pure_iod = qv.Int64Column(default=0)
    obs_in_pure_complete_iod = qv.Int64Column(default=0)
    obs_in_contaminated_iod = qv.Int64Column(default=0)
    obs_as_contaminant_iod = qv.Int64Column(default=0)
    obs_in_mixed_iod = qv.Int64Column(default=0)

    # Orbit determination
    found_pure_od = qv.Int64Column(default=0)
    found_contaminated_od = qv.Int64Column(default=0)
    pure_od = qv.Int64Column(default=0)
    pure_complete_od = qv.Int64Column(default=0)
    contaminated_od = qv.Int64Column(default=0)
    contaminant_od = qv.Int64Column(default=0)
    mixed_od = qv.Int64Column(default=0)
    obs_in_pure_od = qv.Int64Column(default=0)
    obs_in_pure_complete_od = qv.Int64Column(default=0)
    obs_in_contaminated_od = qv.Int64Column(default=0)
    obs_as_contaminant_od = qv.Int64Column(default=0)
    obs_in_mixed_od = qv.Int64Column(default=0)

    # Merge and extend
    found_pure_recovered = qv.Int64Column(default=0)
    found_contaminated_recovered = qv.Int64Column(default=0)
    pure_recovered = qv.Int64Column(default=0)
    pure_complete_recovered = qv.Int64Column(default=0)
    contaminated_recovered = qv.Int64Column(default=0)
    contaminant_recovered = qv.Int64Column(default=0)
    mixed_recovered = qv.Int64Column(default=0)
    obs_in_pure_recovered = qv.Int64Column(default=0)
    obs_in_pure_complete_recovered = qv.Int64Column(default=0)
    obs_in_contaminated_recovered = qv.Int64Column(default=0)
    obs_as_contaminant_recovered = qv.Int64Column(default=0)
    obs_in_mixed_recovered = qv.Int64Column(default=0)


class PipelineSummary(qv.Table):
    test_orbit_id = qv.LargeStringColumn()
    partition_id = qv.LargeStringColumn()
    start_night = qv.Int64Column()
    end_night = qv.Int64Column()
    observations = qv.Int64Column()
    findable = qv.Int64Column(nullable=True)

    # Cluster and link
    found_clusters = qv.Int64Column(default=0)
    completeness_clusters = qv.Float64Column(default=0)
    pure_known_clusters = qv.Int64Column(default=0)
    pure_unknown_clusters = qv.Int64Column(default=0)
    contaminated_clusters = qv.Int64Column(default=0)
    mixed_clusters = qv.Int64Column(default=0)

    # Initial orbit determination
    found_iod = qv.Int64Column(default=0)
    completeness_iod = qv.Float64Column(default=0)
    pure_known_iod = qv.Int64Column(default=0)
    pure_unknown_iod = qv.Int64Column(default=0)
    contaminated_iod = qv.Int64Column(default=0)
    mixed_iod = qv.Int64Column(default=0)

    # Orbit determination
    found_od = qv.Int64Column(default=0)
    completeness_od = qv.Float64Column(default=0)
    pure_known_od = qv.Int64Column(default=0)
    pure_unknown_od = qv.Int64Column(default=0)
    contaminated_od = qv.Int64Column(default=0)
    mixed_od = qv.Int64Column(default=0)

    # Merge and extend
    found_recovered = qv.Int64Column(default=0)
    completeness_recovered = qv.Float64Column(default=0)
    pure_known_recovered = qv.Int64Column(default=0)
    pure_unknown_recovered = qv.Int64Column(default=0)
    contaminated_recovered = qv.Int64Column(default=0)
    mixed_recovered = qv.Int64Column(default=0)


@ray.remote
def analyze_orbit_worker(
    orbit_dir: Union[str, pathlib.Path],
    config: Config,
    labels: ObservationLabels,
    metric: Optional[FindabilityMetric] = None,
    out_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    Ray worker wrapper around analyze_orbit.
    Forces single-threaded difi analysis per orbit for parallelization across orbits.
    """
    return analyze_orbit(
        orbit_dir=orbit_dir,
        config=config,
        labels=labels,
        metric=metric,
        max_processes=1,
        out_dir=out_dir,
    )


def observations_to_analysis_observations(
    observations: Observations,
    labels: ObservationLabels,
) -> AnalysisObservations:
    """
    Convert THOR observations to difi observations format.

    Parameters
    ----------
    observations : Observations
        THOR observations table.
    labels : ObservationLabels
        Observation labels containing object IDs.

    Returns
    -------
    AnalysisObservations
        Observations in difi format.
    """
    logger.info(f"Converting {len(observations)} observations to analysis format")

    cov_matrix = observations.coordinates.covariance.to_matrix()
    ra_sigma = observations.coordinates.sigma_lon
    dec_sigma = observations.coordinates.sigma_lat
    radec_corr = cov_matrix[:, 1, 2] / (ra_sigma * dec_sigma)
    radec_corr = np.where(np.isnan(radec_corr), None, radec_corr)

    # We first create the table with null object IDs since we can't join
    # the coordinate covariances with the labels table.
    analysis_observations = AnalysisObservations.from_kwargs(
        id=observations.id,
        time=observations.coordinates.time,
        ra=observations.coordinates.lon,
        dec=observations.coordinates.lat,
        ra_sigma=ra_sigma,
        dec_sigma=dec_sigma,
        radec_corr=radec_corr,
        mag=observations.photometry.mag,
        mag_sigma=observations.photometry.mag_sigma,
        filter=observations.photometry.filter,
        observatory_code=observations.coordinates.origin.code,
        object_id=None,
        night=observations.night,
    )

    # Filter out the observation labels that are not in the observations table
    labels_filtered = labels.apply_mask(pc.is_in(labels.obs_id, observations.id))
    logger.debug(f"Filtered labels: {len(labels_filtered)} of {len(labels)} labels match observations")

    # Join the observation labels with the analysis observations table
    labels_table = labels_filtered.flattened_table()
    analysis_observations_table = analysis_observations.flattened_table().drop_columns(["object_id"])
    analysis_observations_with_object_id = analysis_observations_table.join(labels_table, ["id"], ["obs_id"])
    analysis_observations_with_object_id = analysis_observations_with_object_id.combine_chunks()

    analysis_observations = AnalysisObservations.from_kwargs(
        id=analysis_observations_with_object_id.column("id"),
        time=Timestamp.from_kwargs(
            days=analysis_observations_with_object_id.column("time.days"),
            nanos=analysis_observations_with_object_id.column("time.nanos"),
            scale=observations.coordinates.time.scale,
        ),
        ra=analysis_observations_with_object_id.column("ra"),
        dec=analysis_observations_with_object_id.column("dec"),
        ra_sigma=analysis_observations_with_object_id.column("ra_sigma"),
        dec_sigma=analysis_observations_with_object_id.column("dec_sigma"),
        radec_corr=analysis_observations_with_object_id.column("radec_corr"),
        mag=analysis_observations_with_object_id.column("mag"),
        mag_sigma=analysis_observations_with_object_id.column("mag_sigma"),
        filter=analysis_observations_with_object_id.column("filter"),
        observatory_code=analysis_observations_with_object_id.column("observatory_code"),
        object_id=analysis_observations_with_object_id.column("object_id"),
        night=analysis_observations_with_object_id.column("night"),
    )
    analysis_observations = qv.defragment(analysis_observations)
    logger.info("Converted observations to analysis format successfully")
    return analysis_observations


def analyze_orbit(
    orbit_dir: Union[str, pathlib.Path],
    config: Config,
    labels: ObservationLabels,
    metric: Optional[FindabilityMetric] = None,
    max_processes: Optional[int] = 1,
    out_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[PipelineAllObjects, PipelineSummary]:
    """
    Analyze a THOR run directory using difi metrics.

    This function reads the filtered observations from a THOR orbit directory,
    determines what is findable according to the given metric, and then analyzes
    each linkage type (clusters, iod_orbits, od_orbits, recovered_orbits) to
    create linkage data products.

    Parameters
    ----------
    orbit_dir : str or pathlib.Path
        Path to the orbit directory containing THOR results.
    config : Config
        THOR configuration object containing min_obs and contamination_percentage
        settings for each linkage type.
    labels : ObservationLabels
        Observation labels mapping observation IDs to object IDs.
    metric : FindabilityMetric, optional
        difi metric to use for determining findability. If None, uses SingletonMetric
        with min_obs and min_nights from config.
    max_processes : int, optional
        Maximum number of processes to use for analysis.

    Returns
    -------
    pipeline_all_objects : PipelineAllObjects
        Table containing analysis results aggregated across all pipeline stages.
    pipeline_partition_summary : PipelineSummary
        Table containing per-partition summaries updated across pipeline stages.
    """
    orbit_dir = pathlib.Path(orbit_dir)
    logger.info(f"Analyzing orbit directory: {orbit_dir}")

    # Use default metric if none provided
    if metric is None:
        metric = SingletonMetric(
            min_obs=config.cluster_min_obs,
            min_nights=config.cluster_min_nights,
        )
        logger.debug(
            f"Using default SingletonMetric with min_obs={config.cluster_min_obs}, min_nights={config.cluster_min_nights}"
        )

    # Read filtered observations
    observations_path = orbit_dir / "filtered_observations.parquet"
    if not observations_path.exists():
        logger.error(f"Filtered observations not found at {observations_path}")
        raise FileNotFoundError(f"Filtered observations not found at {observations_path}")

    logger.info(f"Reading filtered observations from {observations_path}")
    observations = Observations.from_parquet(str(observations_path))
    observations = qv.defragment(observations)
    logger.info(f"Loaded {len(observations)} observations")

    # Convert to difi observations format
    analysis_observations = observations_to_analysis_observations(observations, labels)

    # Run the metric to determine findability
    logger.info("Running findability analysis")
    all_objects, findable_observations, partition_summary = analyze_observations(
        analysis_observations,
        metric=metric,
        by_object=True,
        ignore_after_discovery=False,
        max_processes=max_processes,
    )
    logger.info(f"Found {len(all_objects)} objects, {pc.sum(all_objects.findable).as_py()} findable")

    # Initialize PipelineSummary with base partition information
    test_orbit_id = orbit_dir.name
    pipeline_partition_summary = PipelineSummary.from_kwargs(
        test_orbit_id=[test_orbit_id] * len(partition_summary),
        partition_id=partition_summary.id,
        start_night=partition_summary.start_night,
        end_night=partition_summary.end_night,
        observations=partition_summary.observations,
        findable=partition_summary.findable,
    )

    # Initialize PipelineAllObjects with base findability information
    pipeline_all_objects = PipelineAllObjects.from_kwargs(
        test_orbit_id=[test_orbit_id] * len(all_objects),
        object_id=all_objects.object_id,
        partition_id=all_objects.partition_id,
        mjd_min=all_objects.mjd_min,
        mjd_max=all_objects.mjd_max,
        arc_length=all_objects.arc_length,
        num_obs=all_objects.num_obs,
        num_observations=all_objects.num_obs,
        findable=all_objects.findable,
    )

    # Prepare output directory for this orbit if requested
    orbit_out_dir: Optional[pathlib.Path] = None
    if out_dir is not None:
        orbit_out_dir = pathlib.Path(out_dir) / test_orbit_id
        orbit_out_dir.mkdir(parents=True, exist_ok=True)

    # Define linkage types and their file patterns with corresponding config parameters
    # Format: (linkage_id_col, linkages_file, members_file, linkages_class, linkage_members_class,
    #          min_obs, contamination_percentage, column_suffix)
    linkage_types = {
        "cluster_and_link": (
            "cluster_id",
            "clusters.parquet",
            "cluster_members.parquet",
            FittedClusters,
            FittedClusterMembers,
            config.cluster_min_obs,
            config.iod_contamination_percentage,
            "clusters",
        ),
        "initial_orbit_determination": (
            "orbit_id",
            "iod_orbits.parquet",
            "iod_orbit_members.parquet",
            FittedOrbits,
            FittedOrbitMembers,
            config.iod_min_obs,
            config.iod_contamination_percentage,
            "iod",
        ),
        "orbit_determination": (
            "orbit_id",
            "od_orbits.parquet",
            "od_orbit_members.parquet",
            FittedOrbits,
            FittedOrbitMembers,
            config.od_min_obs,
            config.od_contamination_percentage,
            "od",
        ),
        "merge_and_extend": (
            "orbit_id",
            "recovered_orbits.parquet",
            "recovered_orbit_members.parquet",
            FittedOrbits,
            FittedOrbitMembers,
            config.arc_extension_min_obs,
            config.arc_extension_contamination_percentage,
            "recovered",
        ),
    }

    # Analyze each linkage type
    for linkage_name, (
        linkage_id_col,
        linkages_file,
        members_file,
        linkages_class,
        linkage_members_class,
        min_obs,
        contamination_percentage,
        column_suffix,
    ) in linkage_types.items():
        logger.info(f"Analyzing linkage type: {linkage_name}")
        linkages_path = orbit_dir / linkages_file
        members_path = orbit_dir / members_file

        # Skip if files don't exist
        if not linkages_path.exists() or not members_path.exists():
            logger.warning(f"Skipping {linkage_name}: files not found at {linkages_path} or {members_path}")
            print(f"Skipping {linkage_name}: files not found")
            continue

        # Read linkage members
        # THOR linkage members have columns like "cluster_id" or "orbit_id" and "obs_id"
        logger.debug(f"Reading linkage members from {members_path}")
        linkage_members_thor = linkage_members_class.from_parquet(str(members_path))
        linkage_members = LinkageMembers.from_kwargs(
            linkage_id=linkage_members_thor.table.column(linkage_id_col),
            obs_id=linkage_members_thor.table.column("obs_id"),
        )
        logger.debug(f"Loaded {len(linkage_members)} linkage members")

        # Create partition mapping (assuming all linkages are in a single partition)
        partition_id = partition_summary.id[0].as_py()
        unique_linkage_ids = linkage_members.linkage_id.unique()
        logger.debug(f"Found {len(unique_linkage_ids)} unique linkages")

        partition_mapping = PartitionMapping.from_kwargs(
            linkage_id=unique_linkage_ids,
            partition_id=pa.repeat(partition_id, len(unique_linkage_ids)),
        )

        # Analyze linkages
        logger.debug(
            f"Analyzing linkages with min_obs={min_obs}, contamination_percentage={contamination_percentage}"
        )
        all_objects_updated, all_linkages, partition_summary_updated = analyze_linkages(
            analysis_observations,
            linkage_members,
            all_objects,
            partition_summary=partition_summary,
            partition_mapping=partition_mapping,
            min_obs=min_obs,
            contamination_percentage=contamination_percentage,
        )
        logger.info(f"Linkage analysis complete for {linkage_name}")

        # Merge results into PipelineAllObjects
        # Join on object_id and partition_id to add stage-specific columns
        stage_table = all_objects_updated.flattened_table()
        pipeline_table = pipeline_all_objects.flattened_table()

        # Join the tables
        logger.debug(f"Merging results for {linkage_name}")
        merged_table = pipeline_table.join(
            stage_table.select(
                [
                    "object_id",
                    "partition_id",
                    "found_pure",
                    "found_contaminated",
                    "pure",
                    "pure_complete",
                    "contaminated",
                    "contaminant",
                    "mixed",
                    "obs_in_pure",
                    "obs_in_pure_complete",
                    "obs_in_contaminated",
                    "obs_as_contaminant",
                    "obs_in_mixed",
                ]
            ),
            keys=["object_id", "partition_id"],
        )

        # Update pipeline_all_objects with stage-specific columns
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"found_pure_{column_suffix}",
            merged_table.column("found_pure"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"found_contaminated_{column_suffix}",
            merged_table.column("found_contaminated"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"pure_{column_suffix}",
            merged_table.column("pure"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"pure_complete_{column_suffix}",
            merged_table.column("pure_complete"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"contaminated_{column_suffix}",
            merged_table.column("contaminated"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"contaminant_{column_suffix}",
            merged_table.column("contaminant"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"mixed_{column_suffix}",
            merged_table.column("mixed"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"obs_in_pure_{column_suffix}",
            merged_table.column("obs_in_pure"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"obs_in_pure_complete_{column_suffix}",
            merged_table.column("obs_in_pure_complete"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"obs_in_contaminated_{column_suffix}",
            merged_table.column("obs_in_contaminated"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"obs_as_contaminant_{column_suffix}",
            merged_table.column("obs_as_contaminant"),
        )
        pipeline_all_objects = pipeline_all_objects.set_column(
            f"obs_in_mixed_{column_suffix}",
            merged_table.column("obs_in_mixed"),
        )

        # Update partition summary with stage-specific metrics
        stage_summary_table = partition_summary_updated.flattened_table()
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"found_{column_suffix}", stage_summary_table.column("found")
        )
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"completeness_{column_suffix}", stage_summary_table.column("completeness")
        )
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"pure_known_{column_suffix}", stage_summary_table.column("pure_known")
        )
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"pure_unknown_{column_suffix}", stage_summary_table.column("pure_unknown")
        )
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"contaminated_{column_suffix}", stage_summary_table.column("contaminated")
        )
        pipeline_partition_summary = pipeline_partition_summary.set_column(
            f"mixed_{column_suffix}", stage_summary_table.column("mixed")
        )

        # Save the AllLinkages for this stage if requested
        if orbit_out_dir is not None:
            all_linkages_path = orbit_out_dir / f"all_linkages_{column_suffix}.parquet"
            try:
                all_linkages.to_parquet(str(all_linkages_path))
            except Exception as e:
                logger.warning(f"Failed to write {all_linkages_path}: {e}")

    logger.info(f"Orbit analysis complete for {orbit_dir}")
    # Save per-orbit summaries if requested
    if orbit_out_dir is not None:
        try:
            (orbit_out_dir / "").mkdir(parents=True, exist_ok=True)
            pipeline_all_objects.to_parquet(str(orbit_out_dir / "pipeline_all_objects.parquet"))
        except Exception as e:
            logger.warning(f"Failed to write pipeline_all_objects for {test_orbit_id}: {e}")
        try:
            pipeline_partition_summary.to_parquet(str(orbit_out_dir / "pipeline_partition_summary.parquet"))
        except Exception as e:
            logger.warning(f"Failed to write pipeline_partition_summary for {test_orbit_id}: {e}")
    return pipeline_all_objects, pipeline_partition_summary


def analyze_run(
    run_dir: Union[str, pathlib.Path],
    labels: ObservationLabels,
    config: Optional[Config] = None,
    metric: Optional[FindabilityMetric] = None,
    max_processes: Optional[int] = 1,
    out_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[PipelineAllObjects, PipelineSummary]:
    """
    Analyze an entire THOR run directory containing multiple orbit directories.

    This function iterates through all orbit directories in the run directory
    and analyzes each one using the analyze_orbit function.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Path to the run directory containing multiple orbit subdirectories.
    labels : ObservationLabels
        Observation labels mapping observation IDs to object IDs.
    config : Config, optional
        THOR configuration object. If None, will attempt to load from
        run_dir/inputs/config.json.
    metric : FindabilityMetric, optional
        difi metric to use for determining findability. If None, uses SingletonMetric
        with min_obs and min_nights from config.
    max_processes : int, optional
        Maximum number of processes to use for analysis.

    Returns
    -------
    pipeline_all_objects : PipelineAllObjects
        Aggregated analysis results for all orbits in the run directory.
    pipeline_partition_summary : PipelineSummary
        Aggregated partition summaries for all orbits in the run directory.
    """
    run_dir = pathlib.Path(run_dir)
    logger.info(f"Analyzing run directory: {run_dir}")
    run_out_dir: Optional[pathlib.Path] = None
    if out_dir is not None:
        run_out_dir = pathlib.Path(out_dir)
        run_out_dir.mkdir(parents=True, exist_ok=True)

    # Load config if not provided
    if config is None:
        config_path = run_dir / "inputs" / "config.json"
        if config_path.exists():
            logger.info(f"Loading config from {config_path}")
            config = Config.parse_file(str(config_path))
        else:
            logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                "Please provide a config object or ensure config.json exists in inputs/."
            )

    # Use default metric if none provided
    if metric is None:
        metric = SingletonMetric(
            min_obs=config.cluster_min_obs,
            min_nights=config.cluster_min_nights,
        )
        logger.debug(
            f"Using default SingletonMetric with min_obs={config.cluster_min_obs}, min_nights={config.cluster_min_nights}"
        )

    # Find all orbit directories (exclude inputs directory)
    orbit_dirs = [
        d for d in run_dir.iterdir() if d.is_dir() and d.name != "inputs" and not d.name.startswith(".")
    ]

    if not orbit_dirs:
        logger.error(f"No orbit directories found in {run_dir}")
        raise ValueError(f"No orbit directories found in {run_dir}")

    logger.info(f"Found {len(orbit_dirs)} orbit directories to analyze")

    all_objects_results = []
    partition_summary_results = []

    # Analyze each orbit directory
    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        # Put shared objects in the object store
        config_ref = ray.put(config)
        labels_ref = ray.put(labels)
        metric_ref = ray.put(metric) if metric is not None else None
        out_dir_arg = str(run_out_dir) if run_out_dir is not None else None

        # Launch tasks
        futures = []
        for orbit_dir in sorted(orbit_dirs):
            orbit_name = orbit_dir.name
            logger.info(f"Submitting {orbit_name} to Ray...")
            futures.append(
                analyze_orbit_worker.remote(
                    str(orbit_dir),
                    config_ref,
                    labels_ref,
                    metric_ref,
                    out_dir_arg,
                )
            )

        # Gather results
        for future in futures:
            try:
                pipeline_all_objects, pipeline_partition_summary = ray.get(future)
                all_objects_results.append(pipeline_all_objects)
                partition_summary_results.append(pipeline_partition_summary)
            except FileNotFoundError as e:
                logger.warning(f"One orbit skipped: {e}")
                continue
            except Exception as e:
                logger.error(f"One orbit failed: {e}", exc_info=True)
                continue
    else:
        for orbit_dir in sorted(orbit_dirs):
            orbit_name = orbit_dir.name
            logger.info(f"Analyzing {orbit_name}...")

            try:
                pipeline_all_objects, pipeline_partition_summary = analyze_orbit(
                    orbit_dir=orbit_dir,
                    config=config,
                    labels=labels,
                    metric=metric,
                    max_processes=1,
                    out_dir=run_out_dir,
                )
                all_objects_results.append(pipeline_all_objects)
                partition_summary_results.append(pipeline_partition_summary)
                logger.info(f"{orbit_name} complete")

            except FileNotFoundError as e:
                logger.warning(f"{orbit_name} skipped: {e}")
                continue

            except Exception as e:
                logger.error(f"{orbit_name} failed: {e}", exc_info=True)
                continue

    # Concatenate all results
    if len(all_objects_results) == 0:
        logger.error("No orbits were successfully analyzed")
        raise ValueError("No orbits were successfully analyzed")

    logger.info(f"Concatenating results from {len(all_objects_results)} orbits")
    aggregated_all_objects = qv.concatenate(all_objects_results)
    if aggregated_all_objects.fragmented():
        logger.debug("Defragmenting aggregated all-objects results")
        aggregated_all_objects = qv.defragment(aggregated_all_objects)

    aggregated_partition_summary = qv.concatenate(partition_summary_results)
    if aggregated_partition_summary.fragmented():
        logger.debug("Defragmenting aggregated partition summary results")
        aggregated_partition_summary = qv.defragment(aggregated_partition_summary)

    logger.info(f"Run analysis complete. Total objects: {len(aggregated_all_objects)}")
    return aggregated_all_objects, aggregated_partition_summary
