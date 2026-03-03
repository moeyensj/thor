import os
import shutil

import pyarrow.compute as pc
import pytest
from adam_assist import ASSISTPropagator
from adam_core.utils.helpers import make_observations, make_real_orbits

from ..checkpointing import (
    FittedOrbitMembers,
    FittedOrbits,
    TransformedDetections,
    load_initial_checkpoint_values,
)
from ..clusters import ClusterMembers, Clusters, FittedClusterMembers, FittedClusters
from ..config import Config
from ..main import initialize_use_ray, link_test_orbit
from ..observations import Observations
from ..observations.filters import TestOrbitRadiusObservationFilter
from ..orbit import TestOrbitEphemeris
from ..orbit import TestOrbits as THORbits
from ..range_and_transform import range_and_transform

OBJECT_IDS = [
    "594913 'Aylo'chaxnim (2020 AV2)",
    "163693 Atira (2003 CP20)",
    "(2010 TK7)",
    "3753 Cruithne (1986 TO)",
    "54509 YORP (2000 PH5)",
    "2063 Bacchus (1977 HB)",
    "1221 Amor (1932 EA1)",
    "433 Eros (A898 PA)",
    "3908 Nyx (1980 PA)",
    "434 Hungaria (A898 RB)",
    "1876 Napolitania (1970 BA)",
    "2001 Einstein (1973 EB)",
    "6 Hebe (A847 NA)",
    "6522 Aci (1991 NQ)",
    "10297 Lynnejones (1988 RJ13)",
    "17032 Edlu (1999 FM9)",
    "202930 Ivezic (1998 SG172)",
    "911 Agamemnon (A919 FB)",
    "1143 Odysseus (1930 BH)",
    "1172 Aneas (1930 UA)",
    "3317 Paris (1984 KF)",
    "5145 Pholus (1992 AD)",
    "5335 Damocles (1991 DA)",
    "15760 Albion (1992 QB1)",
    "15788 (1993 SB)",
    "15789 (1993 SC)",
    "1I/'Oumuamua (A/2017 U1)",
]
TOLERANCES = {
    "default": 0.1 / 3600,
    "594913 'Aylo'chaxnim (2020 AV2)": 2 / 3600,
    "1I/'Oumuamua (A/2017 U1)": 5 / 3600,
}

FAILING_OBJECTS = {
    "594913 'Aylo'chaxnim (2020 AV2)": "Fails OD",  # OBJECT_IDS[0]
    "163693 Atira (2003 CP20)": "Recovers 0 orbits",  # OBJECT_IDS[1]
    "(2010 TK7)": "IndexError during pipeline",  # OBJECT_IDS[2]
    "3753 Cruithne (1986 TO)": "Fails OD",  # OBJECT_IDS[3]
    "54509 YORP (2000 PH5)": "Fails OD",  # OBJECT_IDS[4]
    # "2063 Bacchus (1977 HB)": "Fails OD",  # Now passes
    "433 Eros (A898 PA)": "Fails OD",  # OBJECT_IDS[7]
    "3908 Nyx (1980 PA)": "Fails OD",  # OBJECT_IDS[8]
    "5335 Damocles (1991 DA)": "Recovers 0 orbits",  # OBJECT_IDS[22]
    "1I/'Oumuamua (A/2017 U1)": "Fails IOD",
}

RANGE_TRANSFORM_FAILING = {
    "(2010 TK7)": "IndexError in THORbits.from_orbits() due to empty orbit data",
    "3753 Cruithne (1986 TO)": "theta_x values exceed default tolerance",
}


@pytest.fixture
def observations():
    return make_observations()


@pytest.fixture
def orbits():
    return make_real_orbits()


@pytest.fixture
def integration_config(request):
    max_processes = getattr(request, "param", 1)
    config = Config(
        cluster_vx_bins=10,
        cluster_vy_bins=10,
        cluster_vx_min=-0.01,
        cluster_vx_max=0.01,
        cluster_vy_min=-0.01,
        cluster_vy_max=0.01,
        max_processes=max_processes,
        propagator_namespace="adam_assist.ASSISTPropagator",
    )
    return config


@pytest.fixture
def ray_cluster(integration_config):
    import time

    import ray

    if integration_config.max_processes > 1:
        initialize_use_ray(
            num_cpus=integration_config.max_processes,
            object_store_bytes=integration_config.ray_memory_bytes or None,
        )
        # Wait until ray cluster is ready
        while not ray.is_initialized():

            time.sleep(0.1)
    yield
    if integration_config.max_processes > 1:
        ray.shutdown()


def setup_test_data(object_id, orbits, observations, integration_config, max_arc_length=None):
    """
    Selects the observations and orbit for a given object ID and returns the
    test orbit, observations, expected observation IDs and the configuration
    for the integration test.
    """
    orbit = orbits.select("object_id", object_id)
    exposures, detections, associations = observations

    # Select the associations that match this object ID
    associations_i = associations.select("object_id", object_id)
    detections_i = detections.apply_mask(pc.is_in(detections.id, associations_i.detection_id))
    exposures_i = exposures.apply_mask(pc.is_in(exposures.id, detections_i.exposure_id))
    assert len(associations_i) == 90

    if max_arc_length is not None:
        # Limit detections max_arc_length days from the first detection
        time_mask = pc.and_(
            pc.greater_equal(detections_i.time.days, pc.min(detections_i.time.days)),
            pc.less_equal(
                detections_i.time.days,
                pc.min(detections_i.time.days).as_py() + max_arc_length,
            ),
        )
        detections_i = detections_i.apply_mask(time_mask)
        exposures_i = exposures_i.apply_mask(pc.is_in(exposures_i.id, detections_i.exposure_id))
        associations_i = associations_i.apply_mask(pc.is_in(associations_i.detection_id, detections_i.id))

    # Extract the observations that match this object ID
    obs_ids_expected = associations_i.detection_id.unique().sort()

    # Make THOR observations from the detections and exposures
    observations = Observations.from_detections_and_exposures(detections_i, exposures_i)

    if object_id in TOLERANCES:
        integration_config.filter_cell_radius = TOLERANCES[object_id]
    else:
        integration_config.filter_cell_radius = TOLERANCES["default"]

    # Create a test orbit for this object
    test_orbit = THORbits.from_orbits(orbit)

    return test_orbit, observations, obs_ids_expected, integration_config


def test_Orbit_generate_ephemeris_from_observations_empty(orbits):
    # Test that when passed empty observations, TestOrbit.generate_ephemeris_from_observations
    # returns an empty TestOrbitEphemeris
    observations = Observations.empty()
    test_orbit = THORbits.from_orbits(orbits[0])
    result = test_orbit.generate_ephemeris_from_observations(observations, ASSISTPropagator)
    assert len(result) == 0


@pytest.mark.parametrize(
    "object_id",
    [object_id for object_id in OBJECT_IDS if object_id not in RANGE_TRANSFORM_FAILING]
    + [
        pytest.param(object_id, marks=pytest.mark.xfail(reason=reason))
        for object_id, reason in RANGE_TRANSFORM_FAILING.items()
    ],
)
def test_range_and_transform(object_id, orbits, observations, integration_config):
    integration_config.max_processes = 1
    (
        test_orbit,
        observations,
        obs_ids_expected,
        integration_config,
    ) = setup_test_data(object_id, orbits, observations, integration_config)

    if object_id in TOLERANCES:
        integration_config.filter_cell_radius = TOLERANCES[object_id]
    else:
        integration_config.filter_cell_radius = TOLERANCES["default"]
    # Set a filter to include observations within 1 arcsecond of the predicted position
    # of the test orbit

    filters = [TestOrbitRadiusObservationFilter(radius=integration_config.filter_cell_radius)]
    for filter in filters:
        observations, _ = filter.apply(observations, test_orbit, ASSISTPropagator)

    # Run range and transform and make sure we get the correct observations back
    transformed_detections = range_and_transform(
        test_orbit,
        observations,
        propagator_class=ASSISTPropagator,
    )
    # assert len(transformed_detections) == 90
    assert pc.all(
        pc.less_equal(
            pc.abs(transformed_detections.coordinates.theta_x),
            integration_config.filter_cell_radius,
        )
    ).as_py()

    # Ensure we get all the object IDs back that we expect
    obs_ids_actual = transformed_detections.id.unique().sort()
    assert pc.all(pc.equal(obs_ids_actual, obs_ids_expected))


def run_link_test_orbit(test_orbit, observations, config):
    for stage_results in link_test_orbit(test_orbit, observations, config=config):
        if stage_results.name == "recover_orbits":
            recovered_orbits, recovered_orbit_members = stage_results.result
            return recovered_orbits, recovered_orbit_members


@pytest.mark.parametrize(
    "object_id",
    [object_id for object_id in OBJECT_IDS if object_id not in FAILING_OBJECTS.keys()]
    + [
        pytest.param(object_id, marks=pytest.mark.xfail(reason=FAILING_OBJECTS[object_id]))
        for object_id in FAILING_OBJECTS.keys()
    ],
)
@pytest.mark.parametrize("integration_config", [1, 4], indirect=True)
@pytest.mark.integration
def test_link_test_orbit(object_id, orbits, observations, integration_config):
    (
        test_orbit,
        observations,
        obs_ids_expected,
        integration_config,
    ) = setup_test_data(object_id, orbits, observations, integration_config, max_arc_length=14)

    # Run link_test_orbit and make sure we get the correct observations back
    recovered_orbits, recovered_orbit_members = run_link_test_orbit(
        test_orbit, observations, integration_config
    )
    assert len(recovered_orbits) == 1
    assert len(recovered_orbit_members) == len(obs_ids_expected)

    # Ensure we get all the object IDs back that we expect
    obs_ids_actual = recovered_orbit_members.obs_id
    assert pc.all(pc.equal(obs_ids_actual, obs_ids_expected))


@pytest.mark.parametrize("integration_config", [1, 4], indirect=True)
@pytest.mark.benchmark(group="link_test_orbit", min_rounds=5, warmup=True)
def test_benchmark_link_test_orbit(orbits, observations, integration_config, benchmark, ray_cluster):
    object_id = "202930 Ivezic (1998 SG172)"

    (
        test_orbit,
        observations,
        obs_ids_expected,
        integration_config,
    ) = setup_test_data(object_id, orbits, observations, integration_config, max_arc_length=14)

    recovered_orbits, recovered_orbit_members = benchmark(
        run_link_test_orbit, test_orbit, observations, integration_config
    )
    assert len(recovered_orbits) == 1
    assert len(recovered_orbit_members) == len(obs_ids_expected)

    # Ensure we get all the object IDs back that we expect
    obs_ids_actual = recovered_orbit_members.obs_id
    assert pc.all(pc.equal(obs_ids_actual, obs_ids_expected))


@pytest.mark.integration
def test_link_test_orbit_simple(orbits, observations, integration_config):
    """
    Simple integration test: run the full THOR pipeline for a single
    well-behaved object and validate outputs at every stage.
    """
    object_id = "202930 Ivezic (1998 SG172)"
    integration_config.max_processes = 1

    (
        test_orbit,
        observations,
        obs_ids_expected,
        integration_config,
    ) = setup_test_data(object_id, orbits, observations, integration_config, max_arc_length=14)

    stages_seen = []

    for stage_result in link_test_orbit(test_orbit, observations, config=integration_config):
        stages_seen.append(stage_result.name)

        if stage_result.name == "filter_observations":
            (filtered_observations,) = stage_result.result
            assert len(filtered_observations) > 0, "filter_observations produced empty result"

        elif stage_result.name == "generate_ephemeris":
            (test_orbit_ephemeris,) = stage_result.result
            assert len(test_orbit_ephemeris) > 0, "generate_ephemeris produced empty result"

        elif stage_result.name == "range_and_transform":
            (transformed_detections,) = stage_result.result
            assert len(transformed_detections) > 0, "range_and_transform produced empty result"

        elif stage_result.name == "cluster_and_link":
            clusters, cluster_members = stage_result.result
            assert isinstance(clusters, Clusters), "cluster_and_link should return unfitted Clusters"
            assert len(clusters) > 0, "cluster_and_link produced no clusters"
            assert len(cluster_members) > 0, "cluster_and_link produced no cluster members"

        elif stage_result.name == "fit_clusters":
            fitted_clusters, fitted_cluster_members = stage_result.result
            assert isinstance(fitted_clusters, FittedClusters), "fit_clusters should return FittedClusters"
            assert len(fitted_clusters) > 0, "fit_clusters produced no fitted clusters"
            assert len(fitted_cluster_members) > 0, "fit_clusters produced no fitted cluster members"

        elif stage_result.name == "initial_orbit_determination":
            iod_orbits, iod_orbit_members = stage_result.result
            assert len(iod_orbits) > 0, "initial_orbit_determination produced no orbits"
            assert len(iod_orbit_members) > 0, "initial_orbit_determination produced no orbit members"

        elif stage_result.name == "differential_correction":
            od_orbits, od_orbit_members = stage_result.result
            assert len(od_orbits) > 0, "differential_correction produced no orbits"
            assert len(od_orbit_members) > 0, "differential_correction produced no orbit members"

        elif stage_result.name == "recover_orbits":
            recovered_orbits, recovered_orbit_members = stage_result.result
            assert len(recovered_orbits) == 1, f"Expected 1 recovered orbit, got {len(recovered_orbits)}"
            assert len(recovered_orbit_members) == len(
                obs_ids_expected
            ), f"Expected {len(obs_ids_expected)} orbit members, got {len(recovered_orbit_members)}"
            obs_ids_actual = recovered_orbit_members.obs_id
            assert pc.all(pc.equal(obs_ids_actual, obs_ids_expected))

    # Verify all 8 stages executed
    expected_stages = [
        "filter_observations",
        "generate_ephemeris",
        "range_and_transform",
        "cluster_and_link",
        "fit_clusters",
        "initial_orbit_determination",
        "differential_correction",
        "recover_orbits",
    ]
    assert stages_seen == expected_stages, f"Expected stages {expected_stages}, but saw {stages_seen}"


@pytest.fixture
def working_dir():
    path = os.path.join(os.path.dirname(__file__), "data", "checkpoint")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "data", "checkpoint"))


def test_load_initial_checkpoint_values(working_dir):
    # With an empty directly, ensure checkpoint starts at first stage
    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "filter_observations"

    # Create filtered_observations file to simulate first checkpoint
    exposures, detections, associations = make_observations()
    filtered_observations = Observations.from_detections_and_exposures(detections, exposures)
    filtered_observations_path = os.path.join(working_dir, "filtered_observations.parquet")
    filtered_observations.to_parquet(filtered_observations_path)

    checkpoint = load_initial_checkpoint_values(working_dir)

    assert checkpoint.stage == "generate_ephemeris"
    assert len(checkpoint.filtered_observations) == len(filtered_observations)
    assert checkpoint.filtered_observations.coordinates.time.scale == "utc"

    # Create test_orbit_ephemeris file to simulate ephemeris checkpoint
    TestOrbitEphemeris.empty().to_parquet(os.path.join(working_dir, "test_orbit_ephemeris.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "range_and_transform"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.test_orbit_ephemeris is not None

    # Create transformed_detections file to simulate next checkpoint
    TransformedDetections.empty().to_parquet(os.path.join(working_dir, "transformed_detections.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "cluster_and_link"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.transformed_detections is not None

    # Create unfitted clusters files to simulate cluster_and_link checkpoint
    Clusters.empty().to_parquet(os.path.join(working_dir, "clusters.parquet"))
    ClusterMembers.empty().to_parquet(os.path.join(working_dir, "cluster_members.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "fit_clusters"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.clusters is not None
    assert checkpoint.cluster_members is not None

    # Create fitted clusters files to simulate fit_clusters checkpoint
    FittedClusters.empty().to_parquet(os.path.join(working_dir, "fitted_clusters.parquet"))
    FittedClusterMembers.empty().to_parquet(os.path.join(working_dir, "fitted_cluster_members.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "initial_orbit_determination"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.clusters is not None
    assert checkpoint.cluster_members is not None

    # Create iod_orbits file to simulate fourth checkpoint
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "iod_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "iod_orbit_members.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "differential_correction"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.iod_orbits is not None
    assert checkpoint.iod_orbit_members is not None

    # Create od_orbits files to simulate fifth checkpoint
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "od_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "od_orbit_members.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "recover_orbits"
    assert checkpoint.filtered_observations is not None
    assert checkpoint.od_orbits is not None
    assert checkpoint.od_orbit_members is not None

    # Create recovered_orbits files to simulate completed run folder
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "recovered_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "recovered_orbit_members.parquet"))

    checkpoint = load_initial_checkpoint_values(working_dir)
    assert checkpoint.stage == "complete"
    assert checkpoint.recovered_orbits is not None
    assert checkpoint.recovered_orbit_members is not None
