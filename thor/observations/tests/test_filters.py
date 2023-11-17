from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.observations import Exposures, PointSourceDetections
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris
from adam_core.time import Timestamp

from ...config import Config
from ...orbit import TestOrbits
from ..filters import TestOrbitRadiusObservationFilter, filter_observations
from ..observations import Observations


@pytest.fixture
def fixed_test_orbit() -> TestOrbits:
    # An orbit at 1AU going around at (about) 1 degree per day
    coords = CartesianCoordinates.from_kwargs(
        x=[1],
        y=[0],
        z=[0],
        vx=[0],
        vy=[2 * np.pi / 365.25],
        vz=[0],
        time=Timestamp.from_iso8601(["2020-01-01T00:00:00"]),
        origin=Origin.from_kwargs(code=["SUN"]),
        frame="ecliptic",
    )

    return TestOrbits.from_kwargs(
        orbit_id=["test_orbit"],
        coordinates=coords,
    )


@pytest.fixture
def fixed_observers() -> Observers:
    times = Timestamp.from_iso8601(
        [
            "2020-01-01T00:00:00",
            "2020-01-02T00:00:00",
            "2020-01-03T00:00:00",
            "2020-01-04T00:00:00",
            "2020-01-05T00:00:00",
        ]
    )
    return Observers.from_code("I11", times)


@pytest.fixture
def fixed_ephems(fixed_test_orbit: TestOrbits, fixed_observers: Observers) -> Ephemeris:
    return fixed_test_orbit.generate_ephemeris(fixed_observers)


@pytest.fixture
def fixed_exposures(fixed_observers):
    return Exposures.from_kwargs(
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
        times_mjd_utc = np.full(
            N,
            exposure.start_time.rescale("utc").mjd().to_numpy(zero_copy_only=False)[0]
            + exposure.duration.to_numpy(zero_copy_only=False)[0] / 2 / 86400,
        )

        detection_tables.append(
            PointSourceDetections.from_kwargs(
                id=ids,
                exposure_id=exposure_ids,
                time=Timestamp.from_mjd(times_mjd_utc, scale="utc"),
                ra=ra_decs[0].flatten(),
                dec=ra_decs[1].flatten(),
                mag=magnitudes,
            )
        )
    return qv.concatenate(detection_tables)


@pytest.fixture
def fixed_observations(
    fixed_detections: PointSourceDetections, fixed_exposures: Exposures
) -> Observations:
    return Observations.from_detections_and_exposures(fixed_detections, fixed_exposures)


def test_observation_fixtures(fixed_test_orbit, fixed_observations):
    assert len(fixed_test_orbit) == 1
    assert len(pc.unique(fixed_observations.exposure_id)) == 5
    assert len(fixed_observations.coordinates) == 100 * 100 * 5


def test_orbit_radius_observation_filter(fixed_test_orbit, fixed_observations):
    fos = TestOrbitRadiusObservationFilter(
        radius=0.5,
    )
    have = fos.apply(fixed_observations, fixed_test_orbit)
    assert len(pc.unique(have.exposure_id)) == 5
    assert pc.all(
        pc.equal(
            pc.unique(have.exposure_id),
            pc.unique(fixed_observations.exposure_id),
        )
    )
    # Should be about pi/4 fraction of the detections (0.785
    assert len(have.coordinates) < 0.80 * len(fixed_observations.coordinates)
    assert len(have.coordinates) > 0.76 * len(fixed_observations.coordinates)


def test_filter_observations(fixed_observations, fixed_test_orbit):
    # Test that if not filters are passed, we use
    # TestOrbitRadiusObservationFilter by defualt
    config = Config(cell_radius=0.5, max_processes=1)

    have = filter_observations(fixed_observations, fixed_test_orbit, config)
    assert len(pc.unique(have.exposure_id)) == 5
    assert len(have.coordinates) < 0.80 * len(fixed_observations.coordinates)
    assert len(have.coordinates) > 0.76 * len(fixed_observations.coordinates)

    # Make sure if we pass a custom list of filters, they are used
    # instead
    noop_filter = mock.Mock()
    noop_filter.apply.return_value = fixed_observations

    filters = [noop_filter]
    have = filter_observations(
        fixed_observations, fixed_test_orbit, config, filters=filters
    )
    assert len(have.coordinates) == len(fixed_observations.coordinates)

    # Ensure NOOPFilter.apply was run
    filters[0].apply.assert_called_once()
