from thor.utils.spice import useDE440
import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ...testing import testOrbits
from ...utils import getHorizonsVectors
from ...utils import useDE440
from ..state import shiftOrbitsOrigin

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

@useDE440
def test_shiftOrbitsOrigin():
    """
    Query Horizons (via astroquery) for initial state vectors of each target at T0 in the heliocentric
    and barycentric frames, query Horizons for the heliocentric to barycentric vectors. Compare
    these vectors to ones generated using THOR.
    """
    # Grab barycenter to heliocenter vector from Horizons
    horizons_bary_to_helio = getHorizonsVectors(
        ["sun"], 
        T1, 
        location="@ssb",
        aberrations="geometric", 
        id_type="majorbody"
    )
    horizons_bary_to_helio = horizons_bary_to_helio[["x", "y", "z", "vx", "vy", "vz"]].values

    # Grab heliocenter to barycenter vector from Horizons
    horizons_helio_to_bary = getHorizonsVectors(
        ["ssb"], 
        T1, 
        location="@sun", 
        aberrations="geometric", 
        id_type="majorbody"
    )
    horizons_helio_to_bary = horizons_helio_to_bary[["x", "y", "z", "vx", "vy", "vz"]].values

    # Grab barycenter to heliocenter vector from THOR
    thor_helio_to_bary = shiftOrbitsOrigin(
        np.zeros((len(T1), 6), dtype=float), 
        T1, 
        origin_in="barycenter",
        origin_out="heliocenter"
    )

    # Grab heliocenter to barycenter vector from THOR
    thor_bary_to_helio = shiftOrbitsOrigin(
        np.zeros((len(T1), 6), dtype=float), 
        T1, 
        origin_in="heliocenter",
        origin_out="barycenter"
    )

    # Test that the THOR heliocentric to barycentric
    # vector agrees with the Horizons' vector
    # to within the tolerances below
    testOrbits(
        thor_helio_to_bary,
        horizons_helio_to_bary,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    # Test that the THOR barycentric to heliocentric
    # vector agrees with the Horizons' vector
    # to within the tolerances below
    testOrbits(
        thor_bary_to_helio,
        horizons_bary_to_helio,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )
    
    # Grab heliocentric state vector from Horizons for each
    # target at each T1
    horizons_states_helio = getHorizonsVectors(
        TARGETS, 
        T1, 
        location="@sun", 
        aberrations="geometric"
    )
    horizons_states_helio = horizons_states_helio[["x", "y", "z", "vx", "vy", "vz"]].values

    # Grab barycentric state vector from Horizons for each
    # target at each T1
    horizons_states_bary = getHorizonsVectors(
        TARGETS, 
        T1, 
        location="@ssb", 
        aberrations="geometric"
    )
    horizons_states_bary = horizons_states_bary[["x", "y", "z", "vx", "vy", "vz"]].values

    # Stack T1 so each Horizons state has a corresponding time
    T1_ = Time(np.hstack([T1.tdb.mjd for i in range(len(TARGETS))]), scale="tdb", format="mjd")
    
    # Shift heliocentric state to the barycenter
    thor_states_bary = shiftOrbitsOrigin(
        horizons_states_helio,
        T1_,
        origin_in="heliocenter",
        origin_out="barycenter",
    )
    
    # Shift barycentric state to the heliocenter
    thor_states_helio = shiftOrbitsOrigin(
        horizons_states_bary,
        T1_,
        origin_in="barycenter",
        origin_out="heliocenter",
    )
    
    # Test that THOR heliocentric states agree with
    # Horizons' heliocentric states to within the 
    # tolerances below
    testOrbits(
        thor_states_helio,
        horizons_states_helio,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    # Test that THOR barycentric states agree with
    # Horizons' barycentric states to within the 
    # tolerances below
    testOrbits(
        thor_states_bary,
        horizons_states_bary,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    return

def test_shiftOrbitsOrigin_raise():

    with pytest.raises(ValueError):
        # Raise error for incorrect origin_in
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(T1), 6), dtype=float), 
            T1, 
            origin_in="baarycenter",
            origin_out="heliocenter")

        # Raise error for incorrect origin_out
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(T1), 6), dtype=float), 
            T1, 
            origin_in="barycenter",
            origin_out="heeliocenter")

    return
