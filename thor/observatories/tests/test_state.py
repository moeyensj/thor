import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ...utils import testOrbits
from ...utils import getSPICEKernels
from ...utils import getMPCObservatoryCodes
from ...utils import getHorizonsObserverState
from ..state import getObserverState

OBSERVATORIES = ["I11", "005", "F51", "500", "568", "W84", "012", "I40", "286"]

# Set some observation times
TIMES = Time(
    np.arange(54000, 54030, 1), 
    scale="tdb", 
    format="mjd"
)

def test_getObserverStateAgainstHorizons_heliocentric():
    """
    Query Horizons (via astroquery) for heliocentric state vectors of each observatory at each observation time. 
    Use THOR to find heliocentric state vectors of each observatory at each observation time. 
    Compare the resulting state vectors and test how well they agree with the ones pulled from Horizons.
    """
    # Insure SPICE is ready for computation
    getSPICEKernels()

    # Make sure the latest version of the MPC observatory codes
    # has been downloaded
    getMPCObservatoryCodes()
    
    origin = "heliocenter"
   
    # Query Horizons for heliocentric observer states of each observatory at each time
    horizons_states = getHorizonsObserverState(
        OBSERVATORIES, 
        TIMES, 
        origin=origin,
        aberrations="geometric",
    )
    horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

    # Use THOR to find the heliocentric states for each observatory at each time
    observer_states = getObserverState(
        OBSERVATORIES, 
        TIMES, 
        origin=origin,
    )
    observer_states = observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]].values
    
    # Test that each state agrees with Horizons
    # to within the tolerances below
    testOrbits(
        observer_states,
        horizons_states,
        position_tol=(20*u.m),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    return

def test_getObserverStateAgainstHorizons_barycentric():
    """
    Query Horizons (via astroquery) for barycentric state vectors of each observatory at each observation time. 
    Use THOR to find barycentric state vectors of each observatory at each observation time. 
    Compare the resulting state vectors and test how well they agree with the ones pulled from Horizons.
    """
    # Insure SPICE is ready for computation
    getSPICEKernels()

    # Make sure the latest version of the MPC observatory codes
    # has been downloaded
    getMPCObservatoryCodes()
    
    origin = "barycenter"
   
    # Query Horizons for barycentric observer states of each observatory at each time
    horizons_states = getHorizonsObserverState(
        OBSERVATORIES, 
        TIMES, 
        origin=origin,
        aberrations="geometric",
    )
    horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

    # Use THOR to find the barycentric states for each observatory at each time
    observer_states = getObserverState(
        OBSERVATORIES, 
        TIMES, 
        origin=origin,
    )
    observer_states = observer_states[["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]].values
    
    # Test that each state agrees with Horizons
    # to within the tolerances below
    testOrbits(
        observer_states,
        horizons_states,
        position_tol=(20*u.m),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )
    return
        
def test_getObserverState_raises():

    with pytest.raises(ValueError):
        # Raise error for incorrect frame 
        observer_states = getObserverState(["500"], TIMES, frame="eccliptic")
        
        # Raise error for incorrect origin
        observer_states = getObserverState(["500"], TIMES, origin="heeliocenter")

    with pytest.raises(TypeError):
        # Raise error for non-astropy time
        observer_states = getObserverState(["500"], TIMES.tdb.mjd)
