import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time

from ...utils import getHorizonsVectors
from ..state import shiftOrbitsOrigin

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 

TIMES = Time(np.arange(57257.0, 57257.0 + 30, 1), scale="tdb", format="mjd")

def test_shiftOrbitsOriginAgainstHorizons():
    # Grab barycenter to heliocenter vector from Horizons
    horizons_bary_to_helio = getHorizonsVectors("sun", TIMES, location="@ssb", aberrations="geometric", id_type="majorbody")
    horizons_bary_to_helio = horizons_bary_to_helio[["x", "y", "z", "vx", "vy", "vz"]].values

    # Grab heliocenter to barycenter vector from Horizons
    horizons_helio_to_bary = getHorizonsVectors("ssb", TIMES, location="@sun", aberrations="geometric", id_type="majorbody")
    horizons_helio_to_bary = horizons_helio_to_bary[["x", "y", "z", "vx", "vy", "vz"]].values

    # Grab barycenter to heliocenter vector from THOR
    thor_helio_to_bary = shiftOrbitsOrigin(
        np.zeros((len(TIMES), 6), dtype=float), 
        TIMES, 
        origin_in="barycenter",
        origin_out="heliocenter")

    # Grab heliocenter to barycenter vector from THOR
    thor_bary_to_helio = shiftOrbitsOrigin(
        np.zeros((len(TIMES), 6), dtype=float), 
        TIMES, 
        origin_in="heliocenter",
        origin_out="barycenter")

    # Calculate difference between Horizons heliocenter to barycenter vector and THOR's in m
    r_diff_h2b = np.linalg.norm(horizons_helio_to_bary[:, :3] - thor_helio_to_bary[:, :3], axis=1) * u.AU.to(u.m)

    # Calculate difference between Horizons barycenter to heliocenter vector and THOR's in m
    r_diff_b2h = np.linalg.norm(horizons_bary_to_helio[:, :3] - thor_bary_to_helio[:, :3], axis=1) * u.AU.to(u.m)

    # Assert both agree to within 10 cm
    np.testing.assert_allclose(r_diff_h2b,  np.zeros(len(r_diff_h2b)), atol=0.1, rtol=0.0)
    np.testing.assert_allclose(r_diff_b2h,  np.zeros(len(r_diff_b2h)), atol=0.1, rtol=0.0)

    # Calculate difference between Horizons heliocenter to barycenter velocities and THOR's in mm/s
    v_diff_h2b = np.linalg.norm(horizons_helio_to_bary[:, 3:] - thor_helio_to_bary[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)

    # Calculate difference between Horizons barycenter to heliocenter velocities and THOR's in mm/s
    v_diff_b2h = np.linalg.norm(horizons_bary_to_helio[:, 3:] - thor_bary_to_helio[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)

    # Assert both agree to within 1 mm/s
    np.testing.assert_allclose(v_diff_h2b,  np.zeros(len(v_diff_h2b)), atol=1.0, rtol=0.0)
    np.testing.assert_allclose(v_diff_b2h,  np.zeros(len(v_diff_b2h)), atol=1.0, rtol=0.0)

    for target in TARGETS:
        # Grab barycentric state vector from Horizons
        horizons_states_bary = getHorizonsVectors(target, TIMES, location="@ssb", aberrations="geometric")
        horizons_states_bary = horizons_states_bary[["x", "y", "z", "vx", "vy", "vz"]].values
        
        # Grab heliocentric state vector from Horizons
        horizons_states_helio = getHorizonsVectors(target, TIMES, location="@sun", aberrations="geometric")
        horizons_states_helio = horizons_states_helio[["x", "y", "z", "vx", "vy", "vz"]].values
        
        # Shift heliocentric state to the barycenter
        thor_states_bary = shiftOrbitsOrigin(
            horizons_states_helio,
            TIMES,
            origin_in="heliocenter",
            origin_out="barycenter",
        )
        
        # Shift barycentric state to the heliocenter
        thor_states_helio = shiftOrbitsOrigin(
            horizons_states_bary,
            TIMES,
            origin_in="barycenter",
            origin_out="heliocenter",
        )
        
        # Calculate difference between Horizons heliocenter vector and THOR's in m
        r_diff_helio = np.linalg.norm(horizons_states_helio[:, :3] - thor_states_helio[:, :3], axis=1) * u.AU.to(u.m)

        # Calculate difference between Horizons barycenter vector and THOR's in m
        r_diff_bary = np.linalg.norm(horizons_states_bary[:, :3] - thor_states_bary[:, :3], axis=1) * u.AU.to(u.m)

        # Assert both agree to within 10 cm
        np.testing.assert_allclose(r_diff_helio,  np.zeros(len(r_diff_helio)), atol=0.1, rtol=0.0)
        np.testing.assert_allclose(r_diff_bary,  np.zeros(len(r_diff_bary)), atol=0.1, rtol=0.0)

        # Calculate difference between Horizons heliocenter to barycenter velocities and THOR's in mm/s
        v_diff_helio = np.linalg.norm(horizons_states_helio[:, 3:] - thor_states_helio[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)

        # Calculate difference between Horizons barycenter to heliocenter velocities and THOR's in mm/s
        v_diff_bary = np.linalg.norm(horizons_states_bary[:, 3:] - thor_states_bary[:, 3:], axis=1) * (u.AU / u.d).to(u.mm / u.s)

        # Assert both agree to within 1 mm/s
        np.testing.assert_allclose(v_diff_helio,  np.zeros(len(v_diff_helio)), atol=1.0, rtol=0.0)
        np.testing.assert_allclose(v_diff_bary,  np.zeros(len(v_diff_bary)), atol=1.0, rtol=0.0)

    return

def test_shiftOrbitsOrigin_raise():

    with pytest.raises(ValueError):
        # Raise error for incorrect origin_in
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(TIMES), 6), dtype=float), 
            TIMES, 
            origin_in="baarycenter",
            origin_out="heliocenter")

        # Raise error for incorrect origin_out
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(TIMES), 6), dtype=float), 
            TIMES, 
            origin_in="barycenter",
            origin_out="heeliocenter")

    return
