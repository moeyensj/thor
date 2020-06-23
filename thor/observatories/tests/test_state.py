import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ...utils import getHorizonsVectors
from ..state import getObserverState

OBSERVATORIES = ["I11", "I41", "005", "F51", "500", "568", "W84", "012", "I40", "286"]

# Set some observation times
TIMES = Time(np.arange(54000, 54030, 1), scale="tdb", format="mjd")

def test_getObserverStateAgainstHorizons_heliocentric_ecliptic():

    for observatory in OBSERVATORIES:
        # Get state of observer
        observer_states = getObserverState([observatory], TIMES, frame="ecliptic", origin="heliocenter")
        observer_states = observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]].values

        horizons_states = getHorizonsVectors("sun", TIMES, location=observatory, id_type="majorbody")
        horizons_states = -horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Calculate difference in observer position in meters for each epoch
        r_diff = np.linalg.norm(horizons_states[:, :3] - observer_states[:, :3], axis=1) * u.AU.to(u.m)    

        print("Mean difference in heliocentric position vector for observatory {}: {:.3f} m".format(observatory, np.mean(r_diff)))

        # Assert that the heliocentric position is to within 10 meters
        np.testing.assert_allclose(r_diff, np.zeros(len(r_diff)), atol=10., rtol=0.0)

        # Calculate difference in observer velocity in mm/s for each epoch
        v_diff = np.linalg.norm(horizons_states[:, 3:] - observer_states[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)    

        print("Mean difference in heliocentric velocity vector for observatory {}: {:.3f} mm/s".format(observatory, np.mean(v_diff)))

        # Assert that the heliocentric position is to within 1 mm/s
        np.testing.assert_allclose(v_diff, np.zeros(len(v_diff)), atol=1, rtol=0.0)

        return
        
def test_getObserverStateAgainstHorizons_barycentric_ecliptic():

    for observatory in OBSERVATORIES:
        # Get state of observer
        observer_states = getObserverState([observatory], TIMES, frame="ecliptic", origin="barycenter")
        observer_states = observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]].values

        horizons_states = getHorizonsVectors("ssb", TIMES, location=observatory, id_type="majorbody")
        horizons_states = -horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Calculate difference in observer position in meters for each epoch
        r_diff = np.linalg.norm(horizons_states[:, :3] - observer_states[:, :3], axis=1) * u.AU.to(u.m)    

        print("Mean difference in barycentric position vector for observatory {}: {:.3f} m".format(observatory, np.mean(r_diff)))

        # Assert that the barycentric position is to within 10 meters
        np.testing.assert_allclose(r_diff, np.zeros(len(r_diff)), atol=10., rtol=0.0)

        # Calculate difference in observer velocity in mm/s for each epoch
        v_diff = np.linalg.norm(horizons_states[:, 3:] - observer_states[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)    

        print("Mean difference in barycentric velocity vector for observatory {}: {:.3f} mm/s".format(observatory, np.mean(v_diff)))

        # Assert that the barycentric position is to within 1 mm/s
        np.testing.assert_allclose(v_diff, np.zeros(len(v_diff)), atol=1, rtol=0.0)

        return

def test_getObserverState_raises():

    with pytest.raises(ValueError):
        # Raise error for incorrect frame 
        observer_states = getObserverState(["500"], TIMES, frame="eccliptic")

    with pytest.raises(ValueError):
        # Raise error for incorrect origin
        observer_states = getObserverState(["500"], TIMES, origin="heeliocenter")

    with pytest.raises(TypeError):
        # Raise error for non-astropy time
        observer_states = getObserverState(["500"], TIMES.tdb.mjd)
