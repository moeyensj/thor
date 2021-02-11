import numpy as np
from astropy.time import Time
from astropy import units as u

from ...testing import testOrbits
from ..orbits import Orbits
from ..propagate import propagateOrbits

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
EPOCH = 57257.0
DT = np.linspace(0, 30, num=30)
T0 = Time(
    [EPOCH], 
    format="mjd",
    scale="tdb", 
)
T1 = Time(
    EPOCH + DT, 
    format="mjd",
    scale="tdb"
)


def test_propagateOrbits():
    """
    Query Horizons (via astroquery) for initial state vectors of each target at T0, then propagate
    those states to all T1 using THOR's 2-body propagator and OORB's 2-body propagator(via pyoorb).
    Compare the resulting states and test how well they agree.
    """
    # Query Horizons for heliocentric geometric states at each T0
    orbits = Orbits.fromHorizons(
        TARGETS, 
        T0
    )

    # Propagate the state at T0 to all T1 using MJOLNIR 2-body
    states_mjolnir = propagateOrbits(
        orbits, 
        T1, 
        backend="MJOLNIR", 
        backend_kwargs={},
        threads=1,
        chunk_size=1
    )
    states_mjolnir = states_mjolnir[["x", "y", "z", "vx", "vy", "vz"]].values

    # Propagate the state at T0 to all T1 using PYOORB 2-body
    states_pyoorb = propagateOrbits(
        orbits, 
        T1, 
        backend="PYOORB", 
        backend_kwargs={"dynamical_model" : "2"},
        threads=1,
        chunk_size=1
    )
    states_pyoorb = states_pyoorb[["x", "y", "z", "vx", "vy", "vz"]].values

    # Test that the propagated states agree to within the tolerances below
    testOrbits(
       states_mjolnir, 
       states_pyoorb,
       orbit_type="cartesian", 
       position_tol=1*u.mm, 
       velocity_tol=(1*u.mm/u.s), 
       magnitude=True
    )
    return
