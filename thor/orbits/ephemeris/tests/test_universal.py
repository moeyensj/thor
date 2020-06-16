import numpy as np
from astropy.time import Time

from ....constants import Constants as c
from ....utils import _checkTime
from ....utils import getHorizonsVectors
from ....utils import getHorizonsEphemeris
from ..universal import addPlanetaryAberration

MU = c.M_SUN * c.G

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
START_MJD = 57580.0
END_MJD = START_MJD + 30.0
STEP_MJD = 1.0

TIMES = Time(
    np.arange(START_MJD, END_MJD + STEP_MJD, STEP_MJD),
    scale="utc",
    format="mjd"
)

def test_addPlanetaryAberration():

    for target in TARGETS:
        
        # Grab state vectors from HORIZONS
        vectors_horizons = getHorizonsVectors(target, TIMES)
        orbits = vectors_horizons[["x","y","z","vx","vy","vz"]].values
        
        # Grab ephemeris with observer located at the heliocenter
        ephemeris_horizons = getHorizonsEphemeris(target, TIMES, "@sun")
        lt_horizons = ephemeris_horizons["lighttime"].values / (60 * 24)
        
        # Set the observers state vectors to 0 (located at the heliocenter)
        observer_states = np.zeros((len(TIMES), 3), dtype=float)
        
        # Add planetary aberration to the state vector
        orbit_emit, t0_emit, lt = addPlanetaryAberration(
            orbits, 
            TIMES.utc.value,                   
            observer_states,
            lt_tol=1e-10,
            mu=MU,
            max_iter=1000,
            tol=1e-16)
        
        # Assert that the light times agree to within a microsecond between
        # Horizons and THOR
        np.testing.assert_allclose(lt, lt_horizons, atol=1/(24*60*60*1e6))

    return