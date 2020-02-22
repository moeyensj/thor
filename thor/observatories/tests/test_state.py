import numpy as np
from astropy.time import Time
from astropy import units as u

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

def test_getObserverState_observatories():
    R_EARTH_EQUATORIAL = 6378.1363 
    R_EARTH_POLAR = 6356.7523 
    
    observatories = ["I11", "I41", "005", "F51"]
    
    # Set some observation times
    observation_times = Time(np.arange(50000, 59000, 1), scale="tdb", format="mjd")
    
    for observatory in observatories:
        # Get state of observer
        observer_states = getObserverState([observatory], observation_times)
        r_observer = observer_states[["obs_x", "obs_y", "obs_z"]].values

        # Get state of geocenter at the same time
        geocenter_state = getMajorBodyState("earth", observation_times)
        r_geo = geocenter_state[:, :3]

        # Test that the observatory is somewhere between the equatorial radius (+ height of mount everest) and polar radius 
        # of the earth from the geocenter, we could probably increase the accuracy of this test
        # but the ground truth varies with observatory.
        diff = np.linalg.norm(r_observer - r_geo, axis=1) * u.au.to(u.km)
        assert np.all((diff >= R_EARTH_POLAR) & (diff <= R_EARTH_EQUATORIAL + 8.848))
   
    return
    