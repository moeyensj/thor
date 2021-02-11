import numpy as np
from astropy.time import Time
from astropy import units as u

from ...utils import _checkTime
from ...utils import getHorizonsVectors
from ...utils import getHorizonsEphemeris
from ...testing import testEphemeris
from ..orbits import Orbits
from ..ephemeris import generateEphemeris

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
] 
EPOCH = 57257.0
DT = np.array([0])
T0 = Time(
    [EPOCH], 
    format="mjd",
    scale="tdb", 
)
T1 = Time(
    EPOCH + DT, 
    format="mjd",
    scale="tdb"
)
OBSERVATORIES = ["I11", "I41", "005", "F51", "500", "568", "W84", "012", "I40", "286"]
OBSERVERS =  {k:T1 for k in OBSERVATORIES}


def test_generateEphemeris():
    """
    Query Horizons (via astroquery) for initial state vectors of each target at T0, also query Horizons
    for ephemerides for each target as observed by each observatory at T1. Use THOR and PYOORB backend to 
    generate ephemerides for each target as observed by each observatory at T1 using the initial state vectors.
    Compare the resulting ephemerides and test how well they agree with the ones pulled from Horizons.
    """
    # Query Horizons for ephemerides of each target as observed
    # by each observatory
    horizons_ephemeris = getHorizonsEphemeris(
        TARGETS,
        OBSERVERS
    )
    horizons_ephemeris = horizons_ephemeris[["RA", "DEC"]].values

    # Query Horizons for initial state vectors for each target at T0
    horizons_orbits = Orbits.fromHorizons(
        TARGETS,
        T0,
    )

    # Use PYOORB to generate ephemeris for each target observed by 
    # each observer
    pyoorb_ephemeris = generateEphemeris(
        horizons_orbits, 
        OBSERVERS, 
        backend="PYOORB",
    )
    pyoorb_ephemeris = pyoorb_ephemeris[["RA_deg", "Dec_deg"]].values

    # pyoorb's ephemerides agree with Horizons' ephemerides
    # to within the tolerance below.
    testEphemeris(
        pyoorb_ephemeris,
        horizons_ephemeris,
        angle_tol=(50*u.milliarcsecond),
        magnitude=True
    )

    # Use MJOLNIR's 2-body propagator to generate ephemeris for each target observed by 
    # each observer
    mjolnir_ephemeris = generateEphemeris(
        horizons_orbits, 
        OBSERVERS, 
        backend="MJOLNIR",
    )
    mjolnir_ephemeris = mjolnir_ephemeris[["RA_deg", "Dec_deg"]].values

    # MJOLNIR's 2-body ephemerides agree with Horizons' ephemerides
    # to within the tolerance below.
    testEphemeris(
        mjolnir_ephemeris,
        horizons_ephemeris,
        angle_tol=(50*u.milliarcsecond),
        magnitude=True
    )
    return