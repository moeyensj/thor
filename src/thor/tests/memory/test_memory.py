"""
A series of unit tests to run with memray for memory profiling.

# These test should be run with the following command:
pytest -v --memray --native --memray-bin-path=.cache/bench/memory . -m memory

# You can then generate flamegraphs and other reports with something like this:
memray flamegraph .cache/bench/memory/path_to_output.bin

# Open in a browser
open .cache/bench/memory/path_to_output.html

# You can also view a graph of the _system_ memory usage
# for the duration of the test
open .cache.bench/memory/[session_name]/[test_name].png

"""

import os
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
import pytest

TEST_ORBIT_ID = "896831"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONFIG_PROCESSES = [1, 4]


def get_git_branch_or_revision():
    """
    Get the current Git branch name or revision hash
    """
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        if branch != "HEAD":
            return branch
        else:
            # If HEAD is detached, get the revision hash
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except subprocess.CalledProcessError:
        return "unknown"


@pytest.fixture(scope="session")
def git_branch_or_revision():
    return get_git_branch_or_revision()


@pytest.fixture
def memory_snapshot(request, git_branch_or_revision):
    root_path = request.config.getoption("memory_graph_paths", ".cache/bench/memory")
    timestamps = []
    stop_event = threading.Event()  # Event to signal the thread to stop
    mem_usage = []

    def snapshot():
        start_time = time.time()  # Record the start time
        while not stop_event.is_set():
            mem_info = psutil.virtual_memory()
            mem_usage.append(mem_info.used / (1024**2))  # Convert to MB
            timestamps.append(time.time() - start_time)  # Timestamp relative to start
            time.sleep(0.1)

    snapshot_thread = threading.Thread(target=snapshot)
    snapshot_thread.start()
    yield
    stop_event.set()  # Signal the thread to stop
    snapshot_thread.join()  # Wait for the thread to finish

    # Use a style for nicer plots
    plt.style.use("seaborn-v0_8-notebook")

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, mem_usage, color="magenta", linestyle="-", linewidth=2)  # Removed marker
    plt.title(
        f"Memory Usage [{git_branch_or_revision},{TEST_ORBIT_ID} {request.node.name}",
        fontsize=12,
    )
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Memory Usage (MB)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlim(left=0)
    max_mem_usage = max(mem_usage)
    min_mem_usage = min(mem_usage)
    plt.ylim(bottom=min_mem_usage * 0.8, top=max_mem_usage + (max_mem_usage * 0.1))  # Add 10% breathing room

    # Save the plot to a folder based on Git branch/revision in a file based on the test name
    os.makedirs(f"{root_path}/{git_branch_or_revision}", exist_ok=True)
    plt.savefig(f"{root_path}/{git_branch_or_revision}/{TEST_ORBIT_ID}-{request.node.name}.png")


@pytest.fixture
def memory_input_observations():
    from thor.observations import InputObservations

    return InputObservations.from_parquet(FIXTURES_DIR / "input_observations.parquet")


# We are going to test all the major stages used in link_test_orbit
@pytest.fixture
def memory_observations():
    from thor.observations import Observations

    return Observations.from_feather(FIXTURES_DIR / "inputs/observations.feather")


@pytest.fixture
def memory_test_orbit():
    from thor.orbit import TestOrbits

    return TestOrbits.from_parquet(
        FIXTURES_DIR / "inputs/test_orbits.parquet",
        filters=[("orbit_id", "=", TEST_ORBIT_ID)],
    )


@pytest.fixture
def memory_config(request):
    from thor.config import Config

    max_processes = getattr(request, "param", 1)
    return Config(max_processes=max_processes)


@pytest.fixture
def memory_filtered_observations():
    from thor.observations import Observations

    return Observations.from_parquet(FIXTURES_DIR / f"{TEST_ORBIT_ID}/filtered_observations.parquet")


@pytest.fixture
def memory_transformed_detections():
    from thor.range_and_transform import TransformedDetections

    return TransformedDetections.from_parquet(
        FIXTURES_DIR / f"{TEST_ORBIT_ID}/transformed_detections.parquet"
    )


@pytest.fixture
def memory_clusters():
    from thor.clusters import ClusterMembers, Clusters

    clusters = Clusters.from_parquet(FIXTURES_DIR / f"{TEST_ORBIT_ID}/clusters.parquet")
    cluster_members = ClusterMembers.from_parquet(FIXTURES_DIR / f"{TEST_ORBIT_ID}/cluster_members.parquet")
    return clusters, cluster_members


@pytest.fixture
def memory_iod_orbits():
    from thor.orbit_determination import FittedOrbitMembers, FittedOrbits

    iod_orbits = FittedOrbits.from_parquet(FIXTURES_DIR / f"{TEST_ORBIT_ID}/iod_orbits.parquet")
    iod_orbit_members = FittedOrbitMembers.from_parquet(
        FIXTURES_DIR / f"{TEST_ORBIT_ID}/iod_orbit_members.parquet"
    )
    return iod_orbits, iod_orbit_members


@pytest.fixture
def memory_od_orbits():
    from thor.orbit_determination import FittedOrbitMembers, FittedOrbits

    od_orbits = FittedOrbits.from_parquet(FIXTURES_DIR / f"{TEST_ORBIT_ID}/od_orbits.parquet")
    od_orbit_members = FittedOrbitMembers.from_parquet(
        FIXTURES_DIR / f"{TEST_ORBIT_ID}/od_orbit_members.parquet"
    )
    return od_orbits, od_orbit_members


@pytest.fixture
def ray_cluster(memory_config):
    import ray
    from adam_core.ray_cluster import initialize_use_ray

    if memory_config.max_processes > 1:
        initialize_use_ray(num_cpus=memory_config.max_processes, object_store_bytes=4000000000)

    yield
    if memory_config.max_processes > 1:
        ray.shutdown()


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_load_input_observations(memory_snapshot, memory_config, ray_cluster, memory_input_observations):
    from thor.observations import Observations

    # It's always necessary to sort ahead of time, so we include it in our test
    memory_input_observations = memory_input_observations.sort_by(
        ["time.days", "time.nanos", "observatory_code"]
    )
    memory_input_observations = memory_input_observations.set_column(
        "time", memory_input_observations.time.rescale("utc")
    )
    Observations.from_input_observations(memory_input_observations)


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_filter_observations(
    memory_snapshot, memory_config, ray_cluster, memory_observations, memory_test_orbit
):
    from thor.main import filter_observations

    filter_observations(memory_observations, memory_test_orbit, memory_config)


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_range_and_transform(
    memory_snapshot,
    memory_test_orbit,
    memory_filtered_observations,
    memory_config,
    ray_cluster,
):
    from thor.range_and_transform import range_and_transform

    range_and_transform(
        memory_test_orbit,
        memory_filtered_observations,
        max_processes=memory_config.max_processes,
    )


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_cluster_and_link(memory_transformed_detections, memory_config, ray_cluster, memory_snapshot):
    from thor.main import cluster_and_link

    cluster_and_link(
        memory_transformed_detections,
        vx_range=[memory_config.vx_min, memory_config.vx_max],
        vy_range=[memory_config.vy_min, memory_config.vy_max],
        vx_bins=memory_config.vx_bins,
        vy_bins=memory_config.vy_bins,
        radius=memory_config.cluster_radius,
        min_obs=memory_config.cluster_min_obs,
        min_arc_length=memory_config.cluster_min_arc_length,
        alg=memory_config.cluster_algorithm,
        chunk_size=memory_config.cluster_chunk_size,
        max_processes=memory_config.max_processes,
    )


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_initial_orbit_determination(
    memory_config,
    memory_filtered_observations,
    memory_clusters,
    ray_cluster,
    memory_snapshot,
):
    from thor.orbits.iod import initial_orbit_determination

    _, cluster_members = memory_clusters
    initial_orbit_determination(
        memory_filtered_observations,
        cluster_members,
        min_obs=memory_config.iod_min_obs,
        min_arc_length=memory_config.iod_min_arc_length,
        contamination_percentage=memory_config.iod_contamination_percentage,
        rchi2_threshold=memory_config.iod_rchi2_threshold,
        observation_selection_method=memory_config.iod_observation_selection_method,
        # propagator=memory_config.propagator,
        propagator_kwargs={},
        chunk_size=memory_config.iod_chunk_size,
        max_processes=memory_config.max_processes,
    )


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_differential_correction(
    memory_iod_orbits,
    memory_filtered_observations,
    memory_config,
    ray_cluster,
    memory_snapshot,
):
    from thor.orbits.od import differential_correction

    orbits, orbit_members = memory_iod_orbits
    differential_correction(
        orbits,
        orbit_members,
        memory_filtered_observations,
        min_obs=memory_config.od_min_obs,
        min_arc_length=memory_config.od_min_arc_length,
        contamination_percentage=memory_config.od_contamination_percentage,
        rchi2_threshold=memory_config.od_rchi2_threshold,
        delta=memory_config.od_delta,
        max_iter=memory_config.od_max_iter,
        chunk_size=memory_config.od_chunk_size,
        max_processes=memory_config.max_processes,
    )


@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_merge_and_extend(
    memory_od_orbits,
    memory_filtered_observations,
    memory_config,
    ray_cluster,
    memory_snapshot,
):
    from thor.orbits.attribution import merge_and_extend_orbits

    orbits, orbit_members = memory_od_orbits
    merge_and_extend_orbits(
        orbits,
        orbit_members,
        memory_filtered_observations,
        min_obs=memory_config.arc_extension_min_obs,
        min_arc_length=memory_config.arc_extension_min_arc_length,
        contamination_percentage=memory_config.arc_extension_contamination_percentage,
        rchi2_threshold=memory_config.arc_extension_rchi2_threshold,
        radius=memory_config.arc_extension_radius,
        orbits_chunk_size=memory_config.arc_extension_chunk_size,
        max_processes=memory_config.max_processes,
    )


# Disabling until smaller dataset is available
@pytest.mark.memory
@pytest.mark.parametrize("memory_config", CONFIG_PROCESSES, indirect=True)
def test_link_test_orbit(memory_snapshot, memory_config, memory_observations, memory_test_orbit, ray_cluster):
    from thor.main import link_test_orbit

    for result in link_test_orbit(
        memory_test_orbit,
        memory_observations,
        config=memory_config,
    ):
        pass
