import numpy as np
import pandas as pd
from astropy.time import Time

from ..utils import _checkTime
from ..backend import PYOORB
from ..observatories import getObserverState
from .handler import _backendHandler
from .universal_ephemeris import generateEphemerisUniversal


__all__ = [
    "generateEphemeris"
]

def generateEphemeris(orbits, t0, observers, backend="THOR", backend_kwargs=None):
    """
    Generate ephemeris for the orbits and the given observatories. 
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits for which to generate ephemeris. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be 
        expressed in keplerian, cometary or cartesian elements.
    t0 : `~astropy.time.core.Time` (N)
        Epoch at which orbits are defined.
    observers : dict or `~pandas.DataFrame`
        A dictionary with observatory codes as keys and observation_times (`~astropy.time.core.Time`) as values. 
        Or a data frame with observatory codes, observation times (in UTC), and the observer's heliocentric ecliptic state.
        The expected data frame columns are obs_x, obs_y, obs_y and optionally the velocity columns obs_vx, obs_vy, obs_vz.
        If no velocities are not correctly given, then sky-plane velocities will all be zero.
        (See: `~thor.observatories.getObserverState`)
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    ephemeris : `~pandas.DataFrame` (N x M, 21) or (N x M, 18)
        A DataFrame containing the generated ephemeris.
    """
    # Check that t0 is an astropy.time object
    _checkTime(t0, "t0")
    
    observer_states = None
    if type(observers) == dict:
        
        if backend == "THOR":
            observer_states_list = []
            for observatory_code, observation_times in observers.items():
                # Check that the observation times are astropy time objects
                _checkTime(observation_times, "observation_times for observatory {}".format(observatory_code))

                # Get the observer state for observation times and append to list 
                observer_states = getObserverState([observatory_code], observation_times)
                observer_states_list.append(observer_states)

            # Concatenate the dataframes
            observer_states = pd.concat(observer_states_list)
            observer_states.reset_index(inplace=True, drop=True)
        
    elif type(observers) == pd.DataFrame:
        if backend == "PYOORB":
            err = (
                "observers as a `~pandas.DataFrame` is not supported with PYOORB.\n" \
                "Please provide a dictionary with observatory codes as keys and\n" \
                "`~astropy.time.core.Time` objects for observation times as values.\n"
            )
            raise TypeError(err)
        else:
            observer_states = observers

    else:
        err = (
            "observers type not understood.\n" \
            "observers should be one of the following:\n" \
            "  i)  dictionary with observatory codes as keys and\n" \
            "      observation_times (`~astropy.time.core.Time`) as values.\n" \
            "  ii) pandas.DataFrame with observatory codes, observation times (in UTC), and the\n" \
            "      observer's heliocentric ecliptic state. (See: `~thor.observatories.getObserverState`)" 
        )
        raise TypeError(err)

    if backend_kwargs is None:
        backend_kwargs = _backendHandler(backend, "ephemeris")
    
    if backend == "THOR":
        
        ephemeris_dfs = []
        for observatory_code in observer_states["observatory_code"].unique():
            
            observer_selected = observer_states[observer_states["observatory_code"].isin([observatory_code])]
            observation_times = Time(observer_selected["mjd_utc"].values, format="mjd", scale="utc")
            
            # Grab observer state vectors
            cols = ["obs_x", "obs_y", "obs_z"]
            velocity_cols =  ["obs_vx", "obs_vy", "obs_vz"]
            if set(velocity_cols).intersection(set(observer_selected.columns)) == set(velocity_cols):
                observer_selected = observer_selected[cols + velocity_cols].values
            else:
                observer_selected = observer_selected[cols].values
            
            # Generate ephemeris for each orbit 
            ephemeris = generateEphemerisUniversal(
                orbits, 
                t0, 
                observer_selected, 
                observation_times, 
                **backend_kwargs)
            
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
            ephemeris_dfs.append(ephemeris)

        # Concatenate data frames, reset index and then keep only the columns
        # we care about 
        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.reset_index(inplace=True, drop=True)
        ephemeris = ephemeris[[
            "orbit_id",
            "observatory_code",
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
        ]]

    elif backend == "PYOORB":

        PYOORB_CONFIG = {
            "orbit_type" : "cartesian", 
            "magnitude" : 20, 
            "slope" : 0.15, 
            "dynamical_model" : "N",
            "ephemeris_file" : "de430.dat"
        }
        backend = PYOORB(**PYOORB_CONFIG)
        ephemeris = backend.generateEphemeris(
            orbits,
            t0,
            observers
        )

    else: 
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)

    ephemeris.sort_values(
        by=["orbit_id", "observatory_code", "mjd_utc"],
        inplace=True
    )
    ephemeris.reset_index(
        inplace=True,
        drop=True
    )
    return ephemeris

