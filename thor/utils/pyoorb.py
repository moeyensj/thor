import os
import warnings
import numpy as np
import pyoorb as oo

__all__ = [
    "setupPYOORB",
    "_configureOrbitsPYOORB",
    "_configureEpochsPYOORB"
]

def setupPYOORB(ephemeris_file="de430.dat", verbose=False):
    """
    Initialize PYOORB with the designatied JPL ephemeris file.  
    
    Parameters
    ----------
    ephemeris_file : str, optional
        Which JPL ephemeris file to use with PYOORB.
    verbose : bool, optional
        Print progress statements.
    
    Returns
    -------
    None
    """
    if "THOR_PYOORB" in os.environ.keys() and os.environ["THOR_PYOORB"] == "True":
        if verbose:
            print("PYOORB is already enabled.")
    else:
        if verbose:
            print("Enabling PYOORB...")
        if os.environ.get("OORB_DATA") == None:
            os.environ["OORB_DATA"] = os.path.join(os.environ["CONDA_PREFIX"], "share/openorb")
        # Prepare pyoorb
        ephfile = os.path.join(os.getenv('OORB_DATA'), ephemeris_file)
        err = oo.pyoorb.oorb_init(ephfile)
        if err == 0:
            os.environ["THOR_PYOORB"] = "True"
            if verbose:
                print("Done.")
        else:
            warnings.warn("PYOORB returned error code: {}".format(err))
            
    return

def _configureOrbitsPYOORB(orbits, t0, orbit_type="cartesian",  time_scale="TT", magnitude=20, slope=0.15):
    """
    Convert an array of orbits into the format expected by PYOORB. 
    
    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to convert. See orbit_type for expected input format.
    t0 : `~numpy.ndarray` (N)
        Epoch in MJD at which the orbits are defined. 
    orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
        Orbital element representation of the provided orbits. 
        If cartesian:
            x : heliocentric ecliptic J2000 x position in AU
            y : heliocentric ecliptic J2000 y position in AU
            z : heliocentric ecliptic J2000 z position in AU
            vx : heliocentric ecliptic J2000 x velocity in AU per day
            vy : heliocentric ecliptic J2000 y velocity in AU per day
            vz : heliocentric ecliptic J2000 z velocity in AU per day
        If keplerian:
            a : semi-major axis in AU
            e : eccentricity in degrees
            i : inclination in degrees
            Omega : longitude of the ascending node in degrees
            omega : argument of periapsis in degrees
            M0 : mean anomaly in degrees
        If cometary:
            p : perihelion distance in AU
            e : eccentricity in degrees
            i : inclination in degrees
            Omega : longitude of the ascending node in degrees
            omega : argument of periapsis in degrees
            T0 : time of perihelion passage in MJD
    time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
        Time scale of the MJD epochs.
    magnitude : float or `~numpy.ndarray` (N), optional
        Absolute H-magnitude or M1 magnitude. 
    slope : float or `~numpy.ndarray` (N), optional
        Photometric slope parameter G or K1.

    Returns
    -------
    orbits_pyoorb : `~numpy.ndarray` (N, 12)
        Orbits formatted in the format expected by PYOORB. 
            orbit_id : index of input orbits
            elements x6: orbital elements of propagated orbits
            orbit_type : orbit type
            epoch_mjd : epoch of the propagate orbit
            time_scale : time scale of output epochs
            H/M1 : absolute magnitude
            G/K1 : photometric slope parameter
    """
    orbits_ = orbits.copy()
    if orbits_.shape == (6,):
        num_orbits = 1
    else:
        num_orbits = orbits_.shape[0]

    if orbit_type == "cartesian":
        orbit_type = [1 for i in range(num_orbits)]
    elif orbit_type == "cometary":
        orbit_type = [2 for i in range(num_orbits)]
        H = M1
        G = K1
        orbits_[:, 1:5] = np.radians(orbits_[:, 1:5])
    elif orbit_type == "keplerian":
        orbit_type = [3 for i in range(num_orbits)]
        orbits_[:, 1:] = np.radians(orbits_[:, 1:])
    else:
        raise ValueError("orbit_type should be one of {'cartesian', 'keplerian', 'cometary'}")

    if time_scale == "UTC":
        time_scale = [1 for i in range(num_orbits)]
    elif time_scale == "UT1": 
        time_scale = [2 for i in range(num_orbits)]
    elif time_scale == "TT":
        time_scale = [3 for i in range(num_orbits)]
    elif time_scale == "TAI":
        time_scale = [4 for i in range(num_orbits)]
    else:
        raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

    if type(slope) != np.ndarray:
        slope = [slope for i in range(num_orbits)]
    if type(magnitude) != np.ndarray:
        magnitude = [magnitude for i in range(num_orbits)]
    if type(t0) != np.ndarray:
        t0 = [t0 for i in range(num_orbits)]

    ids = [i for i in range(num_orbits)]

    if num_orbits > 1:
        orbits_pyoorb = np.array(
            np.array([
                ids, 
                *list(orbits_.T), 
                 orbit_type, 
                 t0, 
                 time_scale, 
                 magnitude, 
                 slope
            ]).T, 
            dtype=np.double, 
            order='F'
        )
    else:
        orbits_pyoorb = np.array(
            [[
                ids[0], 
                *list(orbits_.T), 
                orbit_type[0], 
                t0[0], 
                time_scale[0], 
                magnitude[0], 
                slope[0]]
            ], 
            dtype=np.double,
            order='F')
        
    return orbits_pyoorb

def _configureEpochsPYOORB(epochs, time_scale):
    """
    Convert an array of orbits into the format expected by PYOORB.
    
    Parameters
    ----------
    epochs : `~numpy.ndarray` (N)
        Epoch in MJD to convert. 
    time_scale : {'UTC', 'UT1', 'TT', 'TAI'} 
        Time scale of the MJD epochs.
        
    Returns
    -------
    epochs_pyoorb : `~numpy.ndarray (N, 2)
        Epochs converted into the PYOORB format.
    """
    num_times = len(epochs)
    if time_scale == "UTC":
        time_scale = [1 for i in range(num_times)]
    elif time_scale == "UT1": 
        time_scale = [2 for i in range(num_times)]
    elif time_scale == "TT":
        time_scale = [3 for i in range(num_times)]
    elif time_scale == "TAI":
        time_scale = [4 for i in range(num_times)]
    else:
        raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")
    
    epochs_pyoorb = np.array(list(np.vstack([epochs, time_scale]).T), dtype=np.double, order='F')
    return epochs_pyoorb
