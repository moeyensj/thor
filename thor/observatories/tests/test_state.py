import numpy as np
from astropy.time import Time

from ...orbits.ephemeris import getMajorBodyState
from ..state import getObserverState

def test_getObserverState_geocenter():
    # Set some observation times
    observation_times = Time(np.arange(50000, 59000, 1), scale="tdb", format="mjd")
    
    # Get state of geocentric 'observer'
    observer_states = getObserverState(["500"], observation_times)
    r_observer = observer_states[["obs_x", "obs_y", "obs_z"]].values
    
    # Get state of geocenter at the same time
    geocenter_state = getMajorBodyState("earth", observation_times)
    r_geo = geocenter_state[:, :3]
    
    # Assert that a geocentric observer is at the exact same location of the geocenter
    np.testing.assert_equal(r_observer, r_geo)
    return
    