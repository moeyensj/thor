import numpy as np
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import useDefaultDEXXX
from ...utils import getHorizonsVectors
from ...utils import getHorizonsObserverState
from ...testing import testOrbits
from ..orbits import Orbits
from ..ephemeris import generateEphemeris
from ..gauss import gaussIOD

TARGETS = [
    "Ivezic",
] 
EPOCH = 59000.0
DT = np.arange(0, 30, 1)
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
SELECTED_OBS = [
    [0, 6, -1],
]

                
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

@useDefaultDEXXX
def test_gaussIOD():

    for target in TARGETS:
        for observatory in OBSERVATORIES:
            for selected_obs in SELECTED_OBS:

                observers = {observatory : T1}

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
                MJOLNIR_KWARGS = {
                    "light_time" : True, 
                    "lt_tol" : 1e-10,
                    "stellar_aberration" : False,
                    "mu" : c.MU,
                    "max_iter" : 1000, 
                    "tol" : 1e-16
                }
                orbits = Orbits(
                    horizons_states[selected_obs[1]:selected_obs[1]+1], 
                    T1[selected_obs[1]:selected_obs[1]+1], 
                    orbit_type="cartesian",
                )
                ephemeris = generateEphemeris(
                    orbits,
                    observers,
                    backend="MJOLNIR",
                    backend_kwargs=MJOLNIR_KWARGS
                )
                coords = ephemeris[["RA_deg", "Dec_deg"]].values
                states = ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values

                # Run IOD
                iod_orbits = gaussIOD(
                    coords[selected_obs, :], 
                    T1.utc.mjd[selected_obs], 
                    observer_states[selected_obs, :3], 
                    velocity_method="gibbs",
                    light_time=True,
                    max_iter=100,
                    iterate=False
                )

                # Select the best IOD orbit
                best_iod_orbit = selectBestIOD(
                    iod_orbits.cartesian, 
                    states[selected_obs[1]:selected_obs[1] + 1]
                )

                # Test that the resulting orbit is within the tolerances of the 
                # true state below
                testOrbits(
                    best_iod_orbit,
                    states[selected_obs[1]:selected_obs[1] + 1],
                    position_tol=(200*u.km),
                    velocity_tol=(1*u.mm/u.s),
                    raise_error=False
                )
    return