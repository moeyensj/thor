import pyarrow.compute as pc
import pytest
from adam_core.utils.helpers import make_observations, make_real_orbits

from ..main_2 import range_and_transform
from ..observations import Observations
from ..observations.filters import TestOrbitRadiusObservationFilter
from ..orbit import TestOrbit as THORbit

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
    "2 Pallas (A802 FA)",
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


@pytest.fixture
def observations():
    return make_observations()


@pytest.fixture
def orbits():
    return make_real_orbits()


def test_Orbit_generate_ephemeris_from_observations_empty(orbits):
    # Test that when passed empty observations, TestOrbit.generate_ephemeris_from_observations
    # returns a Value Error
    observations = Observations.empty()
    test_orbit = THORbit.from_orbits(orbits[0])
    with pytest.raises(ValueError, match="Observations must not be empty."):
        test_orbit.generate_ephemeris_from_observations(observations)


@pytest.mark.parametrize("object_id", OBJECT_IDS)
def test_range_and_transform(object_id, orbits, observations):

    orbit = orbits.select("object_id", object_id)
    exposures, detections, associations = observations

    # Make THOR observations from the detections and exposures
    observations = Observations.from_detections_and_exposures(detections, exposures)

    # Select the associations that match this object ID
    associations_i = associations.select("object_id", object_id)
    assert len(associations_i) == 90

    # Extract the observations that match this object ID
    obs_ids_expected = associations_i.detection_id.unique().sort()

    # Filter the observations to include only those that match this object
    observations = observations.apply_mask(
        pc.is_in(observations.detections.id, obs_ids_expected)
    )

    if object_id in TOLERANCES:
        tolerance = TOLERANCES[object_id]
    else:
        tolerance = TOLERANCES["default"]

    # Set a filter to include observations within 1 arcsecond of the predicted position
    # of the test orbit
    filters = [TestOrbitRadiusObservationFilter(radius=tolerance)]

    # Create a test orbit for this object
    test_orbit = THORbit.from_orbits(orbit)

    # Run range and transform and make sure we get the correct observations back
    transformed_detections = range_and_transform(
        test_orbit, observations, filters=filters
    )
    assert len(transformed_detections) == 90
    assert pc.all(
        pc.less_equal(pc.abs(transformed_detections.coordinates.theta_x), tolerance)
    ).as_py()
    assert pc.all(
        pc.less_equal(pc.abs(transformed_detections.coordinates.theta_y), tolerance)
    ).as_py()

    # Ensure we get all the object IDs back that we expect
    obs_ids_actual = transformed_detections.id.unique().sort()
    assert pc.all(pc.equal(obs_ids_actual, obs_ids_expected))
