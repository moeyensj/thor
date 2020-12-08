import numpy as np
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import getHorizonsVectors
from ...testing import testOrbits
from ..propagate import propagateOrbits

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

# Propagator Configurations
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


def test_propagateOrbits():
    """
    Query Horizons (via astroquery) for initial state vectors of each target at T0, then propagate
    those states to all T1 using THOR's 2-body propagator and OORB's 2-body propagator(via pyoorb).
    Compare the resulting states and test how well they agree.
    """
    # Query Horizons for heliocentric geometric states at each T1
    orbit_cartesian_horizons = getHorizonsVectors(
        TARGETS, 
        T0[:1],
        location="@sun",
        aberrations="geometric"
    )
    orbit_cartesian_horizons = orbit_cartesian_horizons[["x", "y", "z", "vx", "vy", "vz"]].values

    # Propagate the state at T0 to all T1 using THOR 2-body
    states_thor = propagateOrbits(
        orbit_cartesian_horizons, 
        T0, 
        T1, 
        backend="THOR", 
        backend_kwargs=THOR_PROPAGATOR_KWARGS
    )
    states_thor = states_thor[["x", "y", "z", "vx", "vy", "vz"]].values

    # Propagate the state at T0 to all T1 using PYOORB 2-body
    states_pyoorb = propagateOrbits(
        orbit_cartesian_horizons, 
        T0, 
        T1, 
        backend="PYOORB", 
        backend_kwargs=PYOORB_PROPAGATOR_KWARGS
    )
    states_pyoorb = states_pyoorb[["x", "y", "z", "vx", "vy", "vz"]].values

    # Test that the propagated states agree to within the tolerances below
    testOrbits(
       states_thor, 
       states_pyoorb,
       orbit_type="cartesian", 
       position_tol=1*u.mm, 
       velocity_tol=(1*u.mm/u.s), 
       magnitude=True
    )
    return
