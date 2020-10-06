import numpy as np
from astropy.time import Time
from astropy import units as u

from ....constants import Constants as c
from ....utils import testOrbits
from ....utils import getHorizonsObserverState
from ....utils import getHorizonsVectors
from ...ephemeris import generateEphemeris
from ..gauss import gaussIOD

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
] 
EPOCH = 57257.0
DT = np.arange(0, 14)
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
OBSERVATORIES = ["I11", "I41", "500"]

def selectBestIOD(iod_orbits, true_orbit):
    """
    Helper function that selects the best preliminary orbit
    by selecting the orbit closes in position to the 
    true orbit. 
    
    This is intended to only used for testing.
    
    Parameters
    ----------
    iod_orbits : `~numpy.ndarray` (N, 6)
        Cartesian preliminary orbits from IOD functions.
    true_orbit : `~numpy.ndarray` (1, 6)
        True cartesian orbit.
        
    Returns
    -------
    best_iod_orbit : `~numpy.ndarray` (1, 6)
        The orbit closest in position to the true orbit. 
    """
    delta_state = iod_orbits - true_orbit
    delta_position = np.linalg.norm(delta_state[:, :3], axis=1)
    nearest_iod = np.argmin(delta_position)
    
    return iod_orbits[nearest_iod:nearest_iod+1]

def test_gaussIOD_withIterator():

    for target in TARGETS:
        for observatory in OBSERVATORIES:

            observers = {observatory : T1}
            selected_obs = [0, 6, -1]

            # Query Horizons for observatory's state vectors at each T1
            horizons_observer_states = getHorizonsObserverState(
                [observatory], 
                T1, 
                origin="heliocenter", 
                aberrations="geometric"
            )
            observer_states = horizons_observer_states[["x", "y", "z", "vx", "vy", "vz"]].values

            # Query Horizons for target's state vectors at each T1
            horizons_states = getHorizonsVectors(
                [target],
                T1,
                location="@sun",
                aberrations="geometric",
            )
            horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

            # Generate ephemeris using the 2-body integration
            THOR_EPHEMERIS_KWARGS = {
                "light_time" : False, 
                "lt_tol" : 1e-10,
                "stellar_aberration" : False,
                "mu" : c.G * c.M_SUN,
                "max_iter" : 1000, 
                "tol" : 1e-16
            }
            ephemeris = generateEphemeris(
                horizons_states[selected_obs[1]:selected_obs[1]+1], 
                T1[selected_obs[1]:selected_obs[1]+1], 
                observers,
                backend="THOR",
                backend_kwargs=THOR_EPHEMERIS_KWARGS
            )
            coords = ephemeris[["RA_deg", "Dec_deg"]].values
            states = ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values

            # Run IOD
            iod_epochs, iod_orbits = gaussIOD(
                coords[selected_obs, :], 
                T1.utc.mjd[selected_obs], 
                observer_states[selected_obs, :3], 
                velocity_method="gibbs",
                light_time=False,
                max_iter=100,
                iterate=True
            )

            # Select the best IOD orbit
            best_iod_orbit = selectBestIOD(
                iod_orbits, 
                states[selected_obs[1]:selected_obs[1] + 1]
            )

            # Test that the resulting orbit is within the tolerances of the 
            # true state below
            testOrbits(
                best_iod_orbit,
                states[selected_obs[1]:selected_obs[1] + 1],
                position_tol=(100*u.m),
                velocity_tol=(1*u.mm/u.s)
            )
    return