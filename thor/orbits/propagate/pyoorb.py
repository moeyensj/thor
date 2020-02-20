import os
import numpy as np
import pandas as pd
import pyoorb as oo

from ...config import Config
from ...utils import setupPYOORB
from ...utils import _configureOrbitsPYOORB
from ...utils import _configureEpochsPYOORB

__all__ = [
    "propagateOrbitsPYOORB"
]

def propagateOrbitsPYOORB(
        orbits, 
        t0, 
        t1, 
        orbit_type="cartesian", 
        time_scale="TT", 
        magnitude=20, 
        slope=0.15, 
        dynamical_model="N",
        ephemeris_file="de430.dat"):
    """
    Propagate orbits using PYOORB.
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to propagate. See orbit_type for expected input format.
    t0 : `~numpy.ndarray` (N)
        Epoch in MJD at which the orbits are defined. 
    t1 : `~numpy.ndarray` (N)
        Epoch in MJD to which to propagate the orbits. 
    orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
        Heliocentric ecliptic J2000 orbital element representation of the provided orbits
        If 'cartesian':
            x : x-position [AU]
            y : y-position [AU]
            z : z-position [AU]
            vx : x-velocity [AU per day]
            vy : y-velocity [AU per day]
            vz : z-velocity [AU per day]
        If 'keplerian':
            a : semi-major axis [AU]
            e : eccentricity [degrees]
            i : inclination [degrees]
            Omega : longitude of the ascending node [degrees]
            omega : argument of periapsis [degrees]
            M0 : mean anomaly [degrees]
        If 'cometary':
            p : perihelion distance [AU]
            e : eccentricity [degrees]
            i : inclination [degrees]
            Omega : longitude of the ascending node [degrees]
            omega : argument of periapsis [degrees]
            T0 : time of perihelion passage [degrees]
    time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
        Time scale of the MJD epochs.
    magnitude : float or `~numpy.ndarray` (N), optional
        Absolute H-magnitude or M1 magnitude. 
    slope : float or `~numpy.ndarray` (N), optional
        Photometric slope parameter G or K1.
    dynamical_model : {'N', '2'}, optional
        Propagate using N or 2-body dynamics.
    ephemeris_file : str, optional
        Which JPL ephemeris file to use with PYOORB.
        
    Returns
    -------
    propagated : `~pandas.DataFrame`
        Orbits at new epochs.
    """
    setupPYOORB(ephemeris_file=ephemeris_file, verbose=False)
    
    # Convert orbits into PYOORB format
    orbits_pyoorb = _configureOrbitsPYOORB(
        orbits, 
        t0, 
        orbit_type=orbit_type, 
        time_scale=time_scale, 
        magnitude=magnitude, 
        slope=slope
    )
    
    # Convert epochs into PYOORB format
    epochs_pyoorb = _configureEpochsPYOORB(t1, time_scale)
    
    # Propagate orbits to each epoch and append to list 
    # of new states
    states = []
    for epoch in epochs_pyoorb:
        new_state, err = oo.pyoorb.oorb_propagation(
            in_orbits=orbits_pyoorb,
            in_epoch=epoch,
            in_dynmodel=dynamical_model
        )
        states.append(new_state)
    
    # Convert list of new states into a pandas data frame
    if orbit_type == "cartesian":
        elements = ["x", "y", "z", "vx", "vy", "vz"]
    elif orbit_type == "keplerian":
        elements = ["a", "e", "i", "Omega", "omega", "M0"]
    elif orbit_type == "cometary":
        elements = ["q", "e", "i", "Omega", "omega", "T0"]
    else:
        raise ValueError("orbit_type should be one of {'cartesian', 'keplerian', 'cometary'}")
        
    columns = [
        "orbit_id",
        *elements,
        "orbit_type",
        "epoch_mjd",
        "time_scale",
        "H/M1",
        "G/K1"
    ]
    propagated = pd.DataFrame(
        np.concatenate(states),
        columns=columns
    )
    propagated["orbit_id"] = propagated["orbit_id"].astype(int)
    propagated["orbit_type"] = propagated["orbit_type"].astype(int)
    propagated["time_scale"] = propagated["time_scale"].astype(int)
    return propagated
    