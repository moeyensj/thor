import logging
import pathlib
from typing import Literal

import pyarrow.parquet as pq

logger = logging.getLogger("thor")


VALID_STAGES = Literal[
    "filter_observations",
    "generate_ephemeris",
    "range_and_transform",
    "cluster_and_link",
    "initial_orbit_determination",
    "differential_correction",
    "recover_orbits",
    "complete",
]


def detect_checkpoint_stage(test_orbit_directory: pathlib.Path) -> VALID_STAGES:
    """
    Filesystem-based stage detection.

    Looks for existing stage output Parquet files and returns the *next* stage
    that should run.
    """
    # Non-existent directories are treated as "start from scratch". This is
    # important for recursive split runs where some child orbit directories may
    # not have been created yet.
    if not test_orbit_directory.exists():
        logger.info("Working directory does not exist, starting at beginning.")
        return "filter_observations"

    if not test_orbit_directory.is_dir():
        raise ValueError(f"{test_orbit_directory} is not a directory")

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

    if (test_orbit_directory / "clusters.parquet").exists() and (
        test_orbit_directory / "cluster_members.parquet"
    ).exists():
        logger.info("Found clusters, starting stage initial_orbit_determination")
        return "initial_orbit_determination"

    if (test_orbit_directory / "transformed_detections.parquet").exists():
        logger.info("Found transformed detections, starting stage cluster_and_link")
        return "cluster_and_link"

    if (test_orbit_directory / "test_orbit_ephemeris.parquet").exists():
        logger.info("Found test orbit ephemeris, starting stage range_and_transform")
        return "range_and_transform"

    raise ValueError(f"Could not detect stage from {test_orbit_directory}")
