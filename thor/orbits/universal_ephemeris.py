import numpy as np
import pandas as pd
from astropy.time import Time
from numba import jit

from ..constants import Constants as c
from ..coordinates import transformCoordinates
from ..utils import _check_times
from .state import shiftOrbitsOrigin
from .universal_propagate import propagateUniversal
from .aberrations import addLightTime
from .aberrations import addStellarAberration

__all__ = [
    "generateEphemerisUniversal"
]

MU = c.MU

def generateEphemerisUniversal(
        orbits,
        t0,
        observer_states,
        observation_times,
        light_time=True,
        lt_tol=1e-10,
        stellar_aberration=False,
        mu=MU,
        max_iter=1000,
        tol=1e-15
    ):
    """
    Generate ephemeris for orbits relative to the location of the observer.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits for which to generate ephemeris. Orbits should be in heliocentric ecliptic cartesian elements.
    t0 : `~astropy.time.core.Time` (N)
        Epoch at which orbits are defined.
    observer_states : `~numpy.ndarray` (M, 6) or (M, 3)
        State of the observer (optionally, including velocities) at the time of observations.
    observation_times : `~astropy.time.core.Time` (M)
        Observation times at which the observer state vectors are true.
    light_time : bool, optional
        Correct orbits for light travel time.
    lt_tol : float, optional
        Calculate aberration to within this value in time (units of days.)
    stellar_aberration : bool, optional
        Correct for stellar aberration.
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
    # Check that t0 is an astropy.time object
    _check_times(t0, "t0")
    _check_times(observation_times, "observation_times")

    # Propagate orbits to observer states
    propagated_orbits_helio = propagateUniversal(
        orbits,
        t0.tdb.mjd,
        observation_times.tdb.mjd,
        mu=mu,
        max_iter=max_iter,
        tol=tol
    )

    # Stack observation times and observer states (so we can add/subtract arrays later instead of looping)
    observation_times_stacked = Time(
        np.hstack([observation_times.utc.mjd for i in range(len(orbits))]),
        scale="utc",
        format="mjd"
    )
    observer_states_stacked_ = np.vstack([observer_states for i in range(len(orbits))])

    # Check observer_states to see if velocities have been passed
    if observer_states_stacked_.shape[1] == 3:
        observer_states_stacked_helio = np.zeros((len(observer_states_stacked_), 6))
        observer_states_stacked_helio[:, :3] = observer_states_stacked_
    elif observer_states_stacked_.shape[1] == 6:
        observer_states_stacked_helio = observer_states_stacked_
    else:
        err = (
            "observer_states should have shape (M, 3) or (M, 6).\n"
        )
        raise ValueError(err)

    # Shift states to the barycenter
    propagated_orbits_bary = propagated_orbits_helio.copy()
    propagated_orbits_bary[:, 2:] = shiftOrbitsOrigin(
        propagated_orbits_helio[:, 2:],
        observation_times_stacked,
        origin_in="heliocenter",
        origin_out="barycenter"
    )
    observer_states_stacked_bary = shiftOrbitsOrigin(
        observer_states_stacked_helio,
        observation_times_stacked,
        origin_in="heliocenter",
        origin_out="barycenter"
    )

    # Add light time correction
    lt = np.zeros(len(propagated_orbits_helio))
    if light_time is True:

        propagated_orbits_bary_lt, lt = addLightTime(
            propagated_orbits_bary[:, 2:],
            observation_times_stacked.utc.mjd,
            observer_states_stacked_bary[:, :3],
            lt_tol=lt_tol,
            mu=mu,
            max_iter=max_iter,
            tol=tol,
        )
        propagated_orbits_bary[:, 2:] = propagated_orbits_bary_lt

    # Calculate topocentric to target barycentric states [ICRF]
    delta_state_bary = propagated_orbits_bary[:, 2:] - observer_states_stacked_bary

    if stellar_aberration is True:
        delta_state_bary[:, :3] = addStellarAberration(propagated_orbits_bary[:, 2:], observer_states_stacked_bary)

    # Convert topocentric to target state to spherical coordinates
    # including velocities
    state_spherical = transformCoordinates(
        delta_state_bary,
        "ecliptic",
        "equatorial",
        representation_in="cartesian",
        representation_out="spherical"
    )

    # Output results
    ephemeris = np.zeros((len(orbits) * len(observation_times), 21))
    ephemeris[:, 0] = propagated_orbits_helio[:, 0]
    ephemeris[:, 1] = observation_times_stacked.utc.mjd
    ephemeris[:, 2] = state_spherical[:, 1]
    ephemeris[:, 3] = state_spherical[:, 2]
    if observer_states_stacked_.shape[1] == 6:
        ephemeris[:, 4] = state_spherical[:, 4] * np.cos(np.radians(state_spherical[:, 5]))
        ephemeris[:, 5] = state_spherical[:, 5]
    else:
        ephemeris[:, 4] = np.zeros(len(state_spherical), dtype=float)
        ephemeris[:, 5] = np.zeros(len(state_spherical), dtype=float)
    ephemeris[:, 6] = np.linalg.norm(propagated_orbits_helio[:, 2:5], axis=1)
    ephemeris[:, 7] = np.linalg.norm(delta_state_bary[:, :3], axis=1)
    ephemeris[:, 8] = lt
    ephemeris[:, 9:15] = propagated_orbits_helio[:, 2:]
    ephemeris[:, 15:] = observer_states_stacked_helio

    # Make a dataframe with the results
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
    ephemeris.sort_values(
        by=["orbit_id", "mjd_utc"],
        inplace=True
    )
    ephemeris.reset_index(
        inplace=True,
        drop=True
    )
    return ephemeris
