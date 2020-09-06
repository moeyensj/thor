import os
import pprint
import warnings
import numpy as np
import pandas as pd
import pyoorb as oo

from ...config import Config
from ...utils import setupPYOORB
from ...utils import _configureOrbitsPYOORB
from ...utils import _configureEpochsPYOORB

__all__ = [
    "generateEphemerisPYOORB"
]

def generateEphemerisPYOORB(
    orbits,
    t0,
    t1,
    orbit_type="cartesian", 
    time_scale="TT", 
    magnitude=20, 
    slope=0.15, 
    observatory_code="I11",
    dynamical_model="N",
    ephemeris_file="de430.dat"):
    """
    Generate ephemeris using PYOORB.
    
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
    observatory_code : str, optional
        Observatory code for which to generate topocentric ephemeris.
    dynamical_model : {'N', '2'}, optional
        Propagate using N or 2-body dynamics.
    ephemeris_file : str, optional
        Which JPL ephemeris file to use with PYOORB.
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
    
    # Generate ephemeris
    ephemeris, err = oo.pyoorb.oorb_ephemeris_full(
      in_orbits=orbits_pyoorb,
      in_obscode=observatory_code,
      in_date_ephems=epochs_pyoorb,
      in_dynmodel=dynamical_model
    )

    if err == 1:
        warnings.warn("PYOORB has returned an error!", UserWarning)
        with np.printoptions(precision=30, threshold=len(orbits)):
            with open("err.log", "w") as f:
                print("Orbits:", file=f)
                pprint.pprint(orbits, width=140, stream=f)
                print("T0 [MJD TT]:", file=f)
                pprint.pprint(t0.tt.mjd, width=140, stream=f)
                print("T1 [MJD TT]:", file=f)
                pprint.pprint(t1.tt.mjd, width=140, stream=f)
    
    columns = [
        "mjd_utc",
        "RA_deg",
        "Dec_deg",
        "vRAcosDec",
        "vDec",
        "PhaseAngle_deg",
        "SolarElon_deg",
        "r_au",
        "delta_au",
        "VMag",
        "PosAngle_deg",
        "TLon_deg",
        "TLat_deg",
        "TOCLon_deg",
        "TOCLat_deg",
        "HLon_deg",
        "HLat_deg",
        "HOCLon_deg",
        "HOCLat_deg",
        "Alt_deg",
        "SolarAlt_deg",
        "LunarAlt_deg",
        "LunarPhase",
        "LunarElon_deg",
        "obj_x",
        "obj_y",
        "obj_z",
        "obj_vx",
        "obj_vy",
        "obj_vz",
        "obs_x",
        "obs_y",
        "obs_z",
        "TrueAnom"
    ]

    ephemeris = pd.DataFrame(
        np.vstack(ephemeris), 
        columns=columns
    )
    ids = np.arange(0, len(orbits))
    ephemeris["orbit_id"] = [i for i in ids for j in t1]
    ephemeris = ephemeris[["orbit_id"] + columns]
    ephemeris.sort_values(
        by=["orbit_id", "mjd_utc"],
        inplace=True
    )
    ephemeris.reset_index(
        inplace=True,
        drop=True
    )
    return ephemeris