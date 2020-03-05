import numpy as np
import pandas as pd

from ...constants import Constants as c
from ...utils import _checkTime
from ...observatories import getObserverState
from .pyoorb import generateEphemerisPYOORB
from .universal import generateEphemerisUniversal

MU = c.G * c.M_SUN

THOR_EPHEMERIS_KWARGS = {
    "light_time" : True, 
    "lt_tol" : 1e-10,
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-16
}

PYOORB_EPHEMERIS_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "TT", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "2",
    "ephemeris_file" : "de430.dat"
}


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
        (See: `~thor.observatories.getObserverState`)
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    ephemeris : `~pandas.DataFrame` (N x M, 21) or (N x M, 21)
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
        
    elif type(observers) == pd.DataFrame:
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
    
    if backend == "THOR":
        if backend_kwargs == None:
            backend_kwargs = THOR_EPHEMERIS_KWARGS
        
        ephemeris_dfs = []
        for observatory_code in observer_states["observatory_code"].unique():
            
            observer_selected = observer_states[observer_states["observatory_code"] == observatory_code]
            
            # All ephemeris in THOR are done in UTC
            t0_utc = t0.utc.mjd
            observation_times_utc = observer_selected["mjd_utc"].values
            
            # Grab observer state vectors
            observer_states = observer_selected[[
                "obs_x",
                "obs_y",
                "obs_z",
                "obs_vx",
                "obs_vy",
                "obs_vz"]].values
            
            # Generate ephemeris for each orbit 
            ephemeris = generateEphemerisUniversal(
                orbits, 
                t0_utc, 
                observer_states, 
                observation_times_utc, 
                **backend_kwargs)
            
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
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
        if backend_kwargs == None:
            backend_kwargs = PYOORB_EPHEMERIS_KWARGS
            
        ephemeris_dfs = []
        for observatory_code, observation_times in observers.items():

            # Generate ephemeris using PYOORB
            ephemeris = generateEphemerisPYOORB(
                orbits, 
                t0.tt.mjd, 
                observation_times.tt.mjd, 
                observatory_code=observatory_code)

            # Add observatory_code to data frame
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
            ephemeris_dfs.append(ephemeris)

        # Concatenate data frames, reset index and then keep only the columns
        # we care about 
        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.reset_index(inplace=True, drop=True)
        ephemeris = ephemeris[[
            'orbit_id', 
            'observatory_code',
            'mjd_utc', 
            'RA_deg', 
            'Dec_deg', 
            'vRAcosDec',
            'vDec', 
            'r_au', 
            'delta_au', 
            'obj_x',
            'obj_y', 
            'obj_z', 
            'obj_vx', 
            'obj_vy', 
            'obj_vz', 
            'obs_x', 
            'obs_y',
            'obs_z'
        ]]

    else: 
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)
    
    return ephemeris

