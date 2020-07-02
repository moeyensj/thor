import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u

from ....constants import Constants as c
from ....utils import getHorizonsVectors
from ..propagate import propagateOrbits

MU = c.G * c.M_SUN

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 

T0 = Time([57257.0], scale="tdb", format="mjd")
T1 = Time(np.arange(57257.0, 57257.0 + 30, 1), scale="tdb", format="mjd")

THOR_PROPAGATOR_KWARGS = {
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-15,
    "origin" : "heliocenter",
}

PYOORB_PROPAGATOR_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "UTC", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "2",
    "ephemeris_file" : "de430.dat"
}


def test_propagateOrbitsAgainstHorizons():

    for target in TARGETS:
        # Query Horizons for heliocentric geometric states at each T1
        horizons_states = getHorizonsVectors(target, T1, location="@sun", aberrations="geometric")
        horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Propagate the state at T0 to all T1 using THOR 2-body
        thor_states = propagateOrbits(horizons_states[:1], T0, T1, backend="THOR", backend_kwargs=THOR_PROPAGATOR_KWARGS)
        thor_states = thor_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Calculate the difference between THOR's heliocentric position and Horizons in mm
        r_thor_diff = np.linalg.norm(thor_states[:,:3] - horizons_states[:,:3], axis=1) * u.AU.to(u.mm)    

        # Propagate the state at T0 to all T1 using PYOORB 2-body
        pyoorb_states = propagateOrbits(horizons_states[:1, :], T0, T1, backend="PYOORB", backend_kwargs=PYOORB_PROPAGATOR_KWARGS)
        pyoorb_states = pyoorb_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Calculate the difference between PYOORB's heliocentric position and Horizons in mm
        r_pyoorb_diff = np.linalg.norm(pyoorb_states[:,:3] - horizons_states[:,:3], axis=1) * u.AU.to(u.mm)   

        # Assert that PYOORB-Horizons differences and THOR-Horizons differences agree to within 1 mm (both set to 2-body)
        np.testing.assert_allclose(np.abs(r_thor_diff - r_pyoorb_diff), np.zeros(len(r_thor_diff)), atol=1, rtol=0)
        
        # Test that the T1 agrees completely with Horizons since T0 = T1[0]
        np.testing.assert_equal(r_thor_diff[0], 0.0)
        np.testing.assert_equal(r_pyoorb_diff[0], 0.0)