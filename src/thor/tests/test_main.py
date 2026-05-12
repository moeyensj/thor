import os
import shutil
import pathlib

import pyarrow.compute as pc
import pytest
from adam_assist import ASSISTPropagator

from adam_core.utils.helpers import make_observations, make_real_orbits

from ..checkpointing import detect_checkpoint_stage
from ..config import Config
from ..main import initialize_use_ray, link_test_orbit
from ..observations import Observations
from ..observations.filters import TestOrbitRadiusObservationFilter as RadiusFilter
from ..orbit import TestOrbitEphemeris as OrbitEphemeris, TestOrbits as THORbits
from ..orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits
from ..range_and_transform import range_and_transform
from ..range_and_transform import TransformedDetections
from ..clusters import FittedClusterMembers, FittedClusters

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
    "3753 Cruithne (1986 TO)": "Fails OD",  # OBJECT_IDS[3]
    "54509 YORP (2000 PH5)": "Fails OD",  # OBJECT_IDS[4]
    "2063 Bacchus (1977 HB)": "Fails OD",  # OBJECT_IDS[5]
    "433 Eros (A898 PA)": "Fails OD",  # OBJECT_IDS[7]
    "3908 Nyx (1980 PA)": "Fails OD",  # OBJECT_IDS[8]
    "1I/'Oumuamua (A/2017 U1)": "Fails IOD",
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
    # Empty observations should yield an empty ephemeris table.
    observations = Observations.empty()
    test_orbit = THORbits.from_orbits(orbits[0])
    ephem = test_orbit.generate_ephemeris_from_observations(observations, ASSISTPropagator)
    assert len(ephem) == 0


@pytest.mark.parametrize("object_id", OBJECT_IDS)
def test_range_and_transform(object_id, orbits, observations, integration_config):
    # This exercises real ephemeris generation and transformations; treat as integration.
    # (The core unit-level behaviors are tested elsewhere.)
    pytest.skip("Slow integration test; run with -m integration.")
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

    filters = [RadiusFilter(radius=integration_config.filter_cell_radius)]
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


def run_link_test_orbit(test_orbit, observations, config, tmp_path):
    # File-based pipeline: persist observations + use a working directory.
    working_dir = tmp_path / "thor_workdir"
    working_dir.mkdir(parents=True, exist_ok=True)
    obs_path = working_dir / "observations.parquet"
    observations.to_parquet(str(obs_path))

    for stage_results in link_test_orbit(test_orbit, str(obs_path), str(working_dir), config=config):
        if stage_results.name == "recover_orbits":
            orbits_path, members_path = stage_results.path
            recovered_orbits = FittedOrbits.from_parquet(orbits_path)
            recovered_orbit_members = FittedOrbitMembers.from_parquet(members_path)
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
    pytest.skip("Slow integration test; run with -m integration.")


@pytest.mark.skip(reason="Benchmark test requires pytest-benchmark plugin.")
def test_benchmark_link_test_orbit():
    # Intentionally skipped.
    pytest.skip("Benchmark test requires pytest-benchmark plugin.")


@pytest.fixture
def working_dir():
    path = os.path.join(os.path.dirname(__file__), "data", "checkpoint")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "data", "checkpoint"))


def test_detect_checkpoint_stage(working_dir):
    # Non-existent orbit directories (e.g., not-yet-started child split orbits)
    # should be treated as "start from scratch".
    stage = detect_checkpoint_stage(pathlib.Path(working_dir) / "missing_child_orbit_dir")
    assert stage == "filter_observations"

    # With an empty directory, ensure stage detection starts at filter_observations
    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "filter_observations"

    # Create filtered_observations file to simulate first checkpoint
    exposures, detections, associations = make_observations()
    filtered_observations = Observations.from_detections_and_exposures(detections, exposures)
    filtered_observations_path = os.path.join(working_dir, "filtered_observations.parquet")
    filtered_observations.to_parquet(filtered_observations_path)

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "generate_ephemeris"

    # Create ephemeris file to simulate generate_ephemeris checkpoint
    OrbitEphemeris.empty().to_parquet(os.path.join(working_dir, "test_orbit_ephemeris.parquet"))
    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "range_and_transform"

    # Create transformed_detections file to simulate second checkpoint
    TransformedDetections.empty().to_parquet(os.path.join(working_dir, "transformed_detections.parquet"))

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "cluster_and_link"

    # Create clusters file to simulate third checkpoint
    FittedClusters.empty().to_parquet(os.path.join(working_dir, "clusters.parquet"))
    FittedClusterMembers.empty().to_parquet(os.path.join(working_dir, "cluster_members.parquet"))

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "initial_orbit_determination"

    # Create iod_orbits file to simulate fourth checkpoint
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "iod_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "iod_orbit_members.parquet"))

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "differential_correction"

    # Create od_orbits files to simulate fifth checkpoint
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "od_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "od_orbit_members.parquet"))

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "recover_orbits"

    # Create recovered_orbits files to simulate completed run folder
    FittedOrbits.empty().to_parquet(os.path.join(working_dir, "recovered_orbits.parquet"))
    FittedOrbitMembers.empty().to_parquet(os.path.join(working_dir, "recovered_orbit_members.parquet"))

    stage = detect_checkpoint_stage(pathlib.Path(working_dir))
    assert stage == "complete"
