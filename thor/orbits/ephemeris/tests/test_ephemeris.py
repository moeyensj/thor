import numpy as np
from astropy.time import Time
from astropy import units as u

from ....constants import Constants as c
from ....utils import _checkTime
from ....utils import getHorizonsVectors
from ....utils import getHorizonsEphemeris
from ..ephemeris import generateEphemeris

MU = c.G * c.M_SUN

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
OBSERVATORIES = ["I11", "I41", "005", "F51", "500", "568", "W84", "012", "I40", "286"]

T0 = Time([55000.0], scale="tdb", format="mjd")
T1 = Time(np.arange(55000.0, 55000.0 + 1, 1), scale="tdb", format="mjd")
OBSERVERS =  {k:T1 for k in OBSERVATORIES}

THOR_EPHEMERIS_KWARGS = {
    "light_time" : True, 
    "lt_tol" : 1e-16,
    "stellar_aberration" : False,
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-16
}

PYOORB_EPHEMERIS_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "UTC", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "2",
    "ephemeris_file" : "de430.dat"
}


def test_generateEphemerisAgainstHorizons():

    for target in TARGETS:
        # Query Horizons for heliocentric geometric states at each T1
        horizons_states = getHorizonsVectors(target, T0, location="@sun")
        horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values
        
        horizons_topo_states = []
        horizons_ephemeris = []
        for location in OBSERVATORIES:
            # Query Horizons for topocentric geometric states at each T1
            horizons_topo_states_i = getHorizonsVectors(target, T1, location=location, aberrations="geometric")
            horizons_topo_states.append(horizons_topo_states_i[["x", "y", "z", "vx", "vy", "vz"]].values)
            
            horizons_ephemeris_i = getHorizonsEphemeris(target, T1, location)
            horizons_ephemeris.append(horizons_ephemeris_i[["RA", "DEC", "delta", "lighttime"]].values)
        
        horizons_topo_states = np.vstack(horizons_topo_states)
        horizons_ephemeris = np.vstack(horizons_ephemeris)

        # Propagate the state at T0 to all T1 using THOR 2-body
        thor_ephemeris = generateEphemeris(horizons_states, T0, OBSERVERS, backend="THOR", backend_kwargs=THOR_EPHEMERIS_KWARGS)
        thor_ephemeris = thor_ephemeris[["RA_deg", "Dec_deg", "delta_au"]].values
        
        # Calculate the difference between THOR's RA and Dec and Horizons in arcseconds
        radec_thor_diff = np.abs(thor_ephemeris[:,:2] - horizons_ephemeris[:,:2]) * u.deg.to(u.arcsec) 

        # Test that the THOR ephemeris at T0 = T1 is within a 0.1 mas.
        np.testing.assert_allclose(radec_thor_diff, np.zeros((len(radec_thor_diff), 2)), atol=1e-4, rtol=0)
        
        # Calculate the difference between THOR's topocentric distance and Horizons in m
        delta_thor_diff =  np.abs(thor_ephemeris[:,2] - horizons_ephemeris[:,2]) * u.AU.to(u.m)
        
        # Test that the THOR delta at T0 = T1 is within 100 meters of Horizons.
        np.testing.assert_allclose(delta_thor_diff, np.zeros(len(delta_thor_diff)), atol=100, rtol=0)

    return