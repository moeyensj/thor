import numpy as np
import pandas as pd
from numba import jit

from ...constants import Constants as c
from ...coordinates import transformCoordinates
from ..propagate import propagateUniversal

__all__ = [
    "addPlanetaryAberration",
    "generateEphemerisUniversal"
]

MU = c.G * c.M_SUN
C = c.C

@jit(["Tuple((f8[:,:], f8[:], f8[:]))(f8[:,:], f8[:], f8[:,:], f8, f8, i8, f8)"], nopython=True)
def addPlanetaryAberration(orbits, t0, observer_states, lt_tol=1e-10, mu=MU, max_iter=1000, tol=1e-15):
    """
    When generating ephemeris, orbits need to be backwards propagated to the time
    at which the light emitted or relflected from the object towards the observer.
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to correct for planetary aberration.
    t0 : `~numpy.ndarray` (N)
        Epoch at which orbits are defined.
    observer_states : `numpy.ndarray` (N, 3)
        Location of the observer at the time of observeration.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days.)
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation. 
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson 
        method.     

    Returns
    -------
    corrected_orbits : `~numpy.ndarray` (N, 6)
        Orbits adjusted for planetary aberration.
    corrected_t0 : `~numpy.ndarray` (N)
        Aberration adjusted epochs.
    lt : `~numpy.ndarray` (N)
        Light time correction (t0 - corrected_t0).
    """
    corrected_orbits = np.zeros((len(orbits), 6))
    corrected_t0 = np.zeros(len(orbits))
    lts = np.zeros(len(orbits))
    num_orbits = len(orbits)
    for i in range(num_orbits):
        
        # Set up running variables
        orbit_i = orbits[i:i+1, :]
        observer_state_i = observer_states[i:i+1, :]
        t0_i = t0[i:i+1]
        dlt = 1e30
        lt_i = 1e30
     
        j = 0 
        while dlt > lt_tol:
            # Calculate topocentric distance
            rho = np.linalg.norm(orbit_i[:, :3] - observer_state_i)
        
            # Calculate initial guess of light time
            lt = rho / C

            # Calculate difference between previous light time correction 
            # and current guess
            dlt = np.abs(lt - lt_i)

            # Propagate backwards to new epoch
            orbit = propagateUniversal(orbits[i:i+1, :], t0[i:i+1], t0[i:i+1] - lt, mu=mu, max_iter=max_iter, tol=tol)

            # Update running variables
            t0_i = orbit[:, 1]
            orbit_i = orbit[:, 2:]
            lt_i = lt
            
        corrected_orbits[i, :] = orbit[:, 2:]
        corrected_t0[i] = orbit[0, 1]
        lts[i] = lt
        
    return corrected_orbits, corrected_t0, lts

def generateEphemerisUniversal(orbits, t0_utc, observer_states, observation_times_utc, light_time=True, lt_tol=1e-10, mu=MU, max_iter=1000, tol=1e-15):
    """
    Generate ephemeris for orbits relative to the location of the observer. 
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits for which to generate ephemeris.
    t0_utc : `~numpy.ndarray` (N)
        Epoch at which orbits are defined ()
    observer_states : `~numpy.ndarray` (M, 6) or (M, 3)
        State of the observer (optionally, including velocities) at the time of observations.
    observation_times_utc : `~numpy.ndarray` (M)
        Observation times at which the observer state vectors are true.
    light_time : bool, optional
        Correct orbits for light travel time (planetary aberration). 
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days.)
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of 
        AU**3 / d**2. 
    max_iter : int, optional
        Maximum number of iterations over which to converge for propagation. 
    tol : float, optional
        Numerical tolerance to which to compute universal anomaly during propagation using the Newtown-Raphson 
        method.     
    
    Returns
    -------
    ephemeris : `~pandas.DataFrame` (M * N, 21)
        Ephemerides for the each orbit at the desired observation times.
        orbit_id : 
            Index (ID) of orbits in the input orbits array.
        mjd_utc : 
            Time of observation in UTC. 
        RA_deg : 
            Heliocentric J2000 equatorial Right Ascension in degrees.
        Dec_deg : 
            Heliocentric J2000 equatorial Declination in degrees.
        vRAcosDec : 
            Heliocentric J2000 equatorial velocity in Right Ascension in degrees per day. 
        vDec : 
            Heliocentric J2000 equatorial velocity in Declination in degrees per day.
        r_au : 
            Heliocentric distance to object.
        delta_au :
            Topocentric distance to object. 
        light_time : 
            Light time travel time. 
        obj_x : 
            x-position of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obj_y : 
            y-position of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obj_z : 
            z-position of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obj_vx : 
            x-velocity of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obj_vy : 
            y-velocity of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obj_vz : 
            z-velocity of the object at the time of light emission/reflection (if the orbit 
            was adjusted for planetary aberration.)
        obs_x : 
            x-position of the observer at the time of observation.
        obs_y : 
            y-position of the observer at the time of observation.
        obs_z : 
            z-position of the observer at the time of observation.
        obs_vx : 
            x-velocity of the observer at the time of observation.
        obs_vy : 
            y-velocity of the observer at the time of observation.
        obs_vz : 
            z-velocity of the observer at the time of observation.
    """
    # Propagate orbits to observer states
    propagated_orbits = propagateUniversal(orbits, t0_utc, observation_times_utc, mu=mu, max_iter=max_iter, tol=tol)
    
    # Stack observation times and observer states (so we can add/subtract arrays later instead of looping)
    observation_times_utc_stacked = np.hstack([observation_times_utc for i in range(len(orbits))])
    observer_states_stacked_ = np.vstack([observer_states for i in range(len(orbits))])
    
    # Check observer_states to see if velocities have been passed
    if observer_states_stacked_.shape[1] == 3:
        observer_states_stacked = np.zeros((len(observer_states_stacked_), 6))
        observer_states_stacked[:, :3] = observer_states_stacked_   
    elif observer_states_stacked_.shape[1] == 6:
        observer_states_stacked = observer_states_stacked_
    else:
        err = (
            "observer_states should have shape (M, 3) or (M, 6).\n"
        )
        raise ValueError(err)

    # Add light time correction 
    lt = np.zeros(len(propagated_orbits))
    if light_time is True:
        propagated_orbits_lt, t0_lt, lt = addPlanetaryAberration(
            propagated_orbits[:, 2:], 
            observation_times_utc_stacked, 
            observer_states[:, :3], 
            lt_tol=lt_tol,  
            mu=mu, 
            max_iter=max_iter, 
            tol=tol
        )
        propagated_orbits[:, 2:] = propagated_orbits_lt
    
    # Calculate topocentric to target state
    delta_state = propagated_orbits[:, 2:] - observer_states_stacked

    # Convert topocentric to target state to spherical coordinates
    # including velocities
    state_spherical = transformCoordinates(
        delta_state, 
        "ecliptic", 
        "equatorial", 
        representation_in="cartesian", 
        representation_out="spherical"
    )

    # Output results
    ephemeris = np.zeros((len(orbits) * len(observation_times_utc), 21))
    ephemeris[:, 0] = propagated_orbits[:, 0]
    ephemeris[:, 1] = observation_times_utc_stacked
    ephemeris[:, 2] = state_spherical[:, 1]
    ephemeris[:, 3] = state_spherical[:, 2]
    if observer_states_stacked_.shape[1] == 6:
        ephemeris[:, 4] = state_spherical[:, 4] * np.cos(np.radians(state_spherical[:, 5]))
        ephemeris[:, 5] = state_spherical[:, 5] 
    else:
        ephemeris[:, 4] = np.zeros(len(state_spherical), dtype=float)
        ephemeris[:, 5] = np.zeros(len(state_spherical), dtype=float)
    ephemeris[:, 6] = np.linalg.norm(propagated_orbits[:, 2:5], axis=1)
    ephemeris[:, 7] = np.linalg.norm(delta_state[:, :3], axis=1)
    ephemeris[:, 8] = lt
    ephemeris[:, 9:15] = propagated_orbits[:, 2:]
    ephemeris[:, 15:] = observer_states_stacked

    # Make a dataframe with the output results
    ephemeris = pd.DataFrame(
        ephemeris,
        columns=[
            "orbit_id",
            "mjd_utc",
            "RA_deg",
            "Dec_deg",
            "vRAcosDec",
            "vDec",
            "r_au",
            "delta_au",
            "light_time",
            "obj_x",
            "obj_y",
            "obj_z",
            "obj_vx",
            "obj_vy",
            "obj_vz",
            "obs_x",
            "obs_y",
            "obs_z",
            "obs_vx",
            "obs_vy",
            "obs_vz",
        ]
    )
    ephemeris["orbit_id"] = ephemeris["orbit_id"].astype(int)
    return ephemeris
    