import numpy as np
import spiceypy as sp
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import getHorizonsVectors
from ...testing import testOrbits
from ..universal_propagate import propagateUniversal

MU = c.G * c.M_SUN

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
EPOCH = 57257.0
DT = np.linspace(0, 30, num=30)
T0 = Time(
    [EPOCH for i in range(len(TARGETS))], 
    format="mjd",
    scale="tdb", 
)
T1 = Time(
    EPOCH + DT, 
    format="mjd",
    scale="tdb"
)

def test_propagateUniversal():
    """
    Query Horizons (via astroquery) for initial state vectors of each target at T0, then propagate
    those states to all T1 using THOR's 2-body propagator and SPICE's 2-body propagator (via spiceypy).
    Compare the resulting states and test how well they agree.
    """
    # Grab vectors from Horizons at initial epoch
    orbit_cartesian_horizons = getHorizonsVectors(
        TARGETS,
        T1[:1],
        location="@sun",
        aberrations="geometric"
    )
    orbit_cartesian_horizons = orbit_cartesian_horizons[["x", "y", "z", "vx", "vy", "vz"]].values
    
    # Propagate initial states to each T1 using SPICE
    states_spice = []
    for i, target in enumerate(TARGETS): 
        for dt in DT:
            states_spice.append(sp.prop2b(MU, list(orbit_cartesian_horizons[i, :]), dt))
    states_spice = np.array(states_spice)
            
    # Repeat but now using THOR's universal 2-body propagator
    states_thor = propagateUniversal(
        orbit_cartesian_horizons, 
        T0.tdb.mjd, 
        T1.tdb.mjd,  
        mu=MU, 
        max_iter=1000, 
        tol=1e-15
    )

    # Test 2-body propagation using THOR is
    # is within this tolerance of SPICE 2-body
    # propagation
    testOrbits(
       states_thor[:, 2:], 
       states_spice,
       orbit_type="cartesian", 
       position_tol=2*u.cm, 
       velocity_tol=(1*u.mm/u.s), 
       magnitude=True
    )
    return