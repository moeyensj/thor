import astropy.time
import numpy as np
import pyarrow as pa
import pytest
import quivr as qv
from adam_core import coordinates, observers, propagator
from adam_core.observations import detections, exposures

from .. import observation_source, orbit


@pytest.fixture
def fixed_test_orbit():
    # An orbit at 1AU going around at (about) 1 degree per day
    coords = coordinates.CartesianCoordinates.from_kwargs(
        x=[1],
        y=[0],
        z=[0],
        vx=[0],
        vy=[2*np.pi/365.25],
        vz=[0],
        time=coordinates.Times.from_astropy(astropy.time.Time("2020-01-01T00:00:00")),
        origin=coordinates.Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    return orbit.TestOrbit(
        coordinates=coords,
        orbit_id="test_orbit",
    )


@pytest.fixture
def fixed_observers():
    times = astropy.time.Time(
        [
            "2020-01-01T00:00:00",
            "2020-01-02T00:00:00",
            "2020-01-03T00:00:00",
            "2020-01-04T00:00:00",
            "2020-01-05T00:00:00",
        ]
    )
    return observers.Observers.from_code("I11", times)


@pytest.fixture
def fixed_ephems(fixed_test_orbit, fixed_observers):
    prop = propagator.PYOORB()
    return prop.generate_ephemeris(fixed_test_orbit.orbit, fixed_observers).left_table


@pytest.fixture
def fixed_exposures(fixed_observers):
    return exposures.Exposures.from_kwargs(
        id=[str(i) for i in range(len(fixed_observers))],
        start_time=fixed_observers.coordinates.time,
        duration=[30 for i in range(len(fixed_observers))],
        filter=["i" for i in range(len(fixed_observers))],
        observatory_code=fixed_observers.code,
    )


@pytest.fixture
def fixed_detections(fixed_ephems, fixed_exposures):
    # Return PointSourceDetections which form a 100 x 100 grid in
    # RA/Dec, evenly spanning 1 square degree, for each exposure
    detection_tables = []
    for ephem, exposure in zip(fixed_ephems, fixed_exposures):
        ra_center = ephem.coordinates.lon[0].as_py()
        dec_center = ephem.coordinates.lat[0].as_py()

        ras = np.linspace(ra_center - 0.5, ra_center + 0.5, 100)
        decs = np.linspace(dec_center - 0.5, dec_center + 0.5, 100)

        ra_decs = np.meshgrid(ras, decs)

        N = len(ras) * len(decs)
        ids = [str(i) for i in range(N)]
        exposure_ids = pa.concat_arrays([exposure.id] * N)
        magnitudes = [20] * N

        detection_tables.append(
            detections.PointSourceDetections.from_kwargs(
                id=ids,
                exposure_id=exposure_ids,
                ra=ra_decs[0].flatten(),
                dec=ra_decs[1].flatten(),
                mag=magnitudes,
            )
        )
    return qv.concatenate(detection_tables)


@pytest.fixture
def fixed_observations(fixed_detections, fixed_exposures):
    return observation_source.Observations(fixed_detections, fixed_exposures)


def test_observation_fixtures(fixed_test_orbit, fixed_observations):
    assert len(fixed_test_orbit.orbit) == 1
    assert len(fixed_observations.exposures) == 5
    assert len(fixed_observations.detections) == 100 * 100 * 5


def test_static_observation_source(fixed_test_orbit, fixed_observations):
    sos = observation_source.StaticObservationSource(observations=fixed_observations)
    have = sos.gather_observations(fixed_test_orbit)

    assert have == fixed_observations


def test_fixed_radius_observation_source(fixed_test_orbit, fixed_observations):
    fos = observation_source.FixedRadiusObservationSource(
        radius=0.5,
        all_observations=fixed_observations,
    )
    have = fos.gather_observations(fixed_test_orbit)
    assert len(have.exposures) == 5
    assert have.exposures == fixed_observations.exposures
    # Should be about pi/4 fraction of the detections (0.785
    assert len(have.detections) < 0.80 * len(fixed_observations.detections)
    assert len(have.detections) > 0.76 * len(fixed_observations.detections)

