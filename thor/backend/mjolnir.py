import os
import warnings
import subprocess 
import numpy as np
import pandas as pd
from astropy.time import Time

from ..constants import Constants as c
from ..utils import _checkTime
from ..orbits import propagateUniversal
from ..orbits import generateEphemerisUniversal
from ..orbits import shiftOrbitsOrigin
from ..observatories import getObserverState
from .backend import Backend

MU = c.G * c.M_SUN
MJOLNIR_CONFIG = {
    "origin" : "heliocenter",
    "light_time" : True, 
    "lt_tol" : 1e-10,
    "stellar_aberration" : False,
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-16 
}

class MJOLNIR(Backend):
    
    def __init__(self, **kwargs):
        
        # Make sure only the correct kwargs
        # are passed to the constructor
        allowed_kwargs = MJOLNIR_CONFIG.keys()
        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError()
        
        # If an allowed kwarg is missing, add the 
        # default 
        for k in allowed_kwargs:
            if k not in kwargs:
                kwargs[k] = MJOLNIR_CONFIG[k]
        
        super(MJOLNIR, self).__init__(**kwargs)

        return

    def _propagateOrbits(self, orbits, t1):
        """
        

        """
        # All propagations in THOR should be done with times in the TDB time scale
        t0_tdb = orbits.epochs.tdb.mjd
        t1_tdb = t1.tdb.mjd

        if self.origin == "barycenter":
            # Shift orbits to barycenter
            orbits_ = shiftOrbitsOrigin(
                orbits.cartesian, 
                orbits.epochs,  
                origin_in="heliocenter",
                origin_out="barycenter"
            )

        elif self.origin == "heliocenter":
            orbits_ = orbits.cartesian
            
        else:
            err = (
                "origin should be one of {'heliocenter', 'barycenter'}"
            )
            raise ValueError(err)

        propagated = propagateUniversal(
            orbits_, 
            t0_tdb, 
            t1_tdb, 
            mu=self.mu,
            max_iter=self.max_iter,
            tol=self.tol
        )

        if self.origin == "barycenter":
            t1_tdb_stacked = Time(
                propagated[:, 1], 
                scale="tdb", 
                format="mjd"
            )
            propagated[:, 2:] = shiftOrbitsOrigin(
                propagated[:, 2:], 
                t1_tdb_stacked, 
                origin_in="barycenter",
                origin_out="heliocenter"
            )

        propagated = pd.DataFrame(
            propagated,
            columns=[
                "orbit_id",
                "epoch_mjd_tdb",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
            ]
        )
        propagated["orbit_id"] = propagated["orbit_id"].astype(int)

        if orbits.ids is not None:
            propagated["orbit_id"] = orbits.ids[propagated["orbit_id"].values]

        return propagated

    def _generateEphemeris(self, orbits, observers):

        observer_states_list = []
        for observatory_code, observation_times in observers.items():
            # Check that the observation times are astropy time objects
            _checkTime(
                observation_times, 
                "observation_times for observatory {}".format(observatory_code)
            )

            # Get the observer state for observation times and append to list 
            observer_states = getObserverState(
                [observatory_code], 
                observation_times
            )
            observer_states_list.append(observer_states)

        # Concatenate the dataframes
        observer_states = pd.concat(observer_states_list)
        observer_states.reset_index(
            inplace=True, 
            drop=True
        )

        ephemeris_dfs = []
        for observatory_code in observer_states["observatory_code"].unique():
            
            observer_selected = observer_states[observer_states["observatory_code"].isin([observatory_code])]
            observation_times = observers[observatory_code]
            
            # Grab observer state vectors
            cols = ["obs_x", "obs_y", "obs_z"]
            velocity_cols =  ["obs_vx", "obs_vy", "obs_vz"]
            if set(velocity_cols).intersection(set(observer_selected.columns)) == set(velocity_cols):
                observer_selected = observer_selected[cols + velocity_cols].values
            else:
                observer_selected = observer_selected[cols].values
            
            # Generate ephemeris for each orbit 
            ephemeris = generateEphemerisUniversal(
                orbits.cartesian, 
                orbits.epochs,
                observer_selected, 
                observation_times, 
                light_time=self.light_time, 
                lt_tol=self.lt_tol, 
                stellar_aberration=self.stellar_aberration, 
                mu=self.mu, 
                max_iter=self.max_iter, 
                tol=self.tol
            )
            
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
            ephemeris_dfs.append(ephemeris)

        # Concatenate data frames, reset index and then keep only the columns
        # we care about 
        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.reset_index(
            inplace=True, 
            drop=True
        )
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

        if orbits.ids is not None:
            ephemeris["orbit_id"] = orbits.ids[ephemeris["orbit_id"].values]
        return ephemeris
