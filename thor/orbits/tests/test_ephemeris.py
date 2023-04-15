import os

import pandas as pd
from astropy import units as u
from astropy.time import Time

from ...testing import testEphemeris
from ..ephemeris import generateEphemeris
from ..orbits import Orbits

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../testing/data"
)


def test_generateEphemeris():
    """
    Read the test data set for initial state vectors of each target at t0, and read the test data set
    for ephemerides for each target as observed by each observatory at t1. Use PYOORB backend to
    generate ephemerides for each target as observed by each observatory at t1 using the initial state vectors.
    Compare the resulting ephemerides and test how well they agree with the ones pulled from Horizons.
    """
    # Read vectors from test data set
    vectors_df = pd.read_csv(os.path.join(DATA_DIR, "vectors.csv"))

    # Read ephemerides from test data
    ephemeris_df = pd.read_csv(os.path.join(DATA_DIR, "ephemeris.csv"))

    # Limit vectors and ephemerides to elliptical orbits only
    elliptical_vectors = (~vectors_df["orbit_class"].str.contains("Hyperbolic")) & (
        ~vectors_df["orbit_class"].str.contains("Parabolic")
    )
    vectors_df = vectors_df[elliptical_vectors]

    elliptical_ephemeris = ephemeris_df["targetname"].isin(
        vectors_df["targetname"].unique()
    )
    ephemeris_df = ephemeris_df[elliptical_ephemeris]

    # Get the target names
    targets = vectors_df["targetname"].unique()

    # Get the initial epochs
    t0 = Time(vectors_df["mjd_tdb"].values, scale="tdb", format="mjd")

    # Pull state vectors
    vectors = vectors_df[["x", "y", "z", "vx", "vy", "vz"]].values

    # Create orbits class
    orbits = Orbits(vectors, t0, ids=targets)

    # Construct observers' dictionary
    observers = {}
    for observatory_code in ephemeris_df["observatory_code"].unique():
        observers[observatory_code] = Time(
            ephemeris_df[ephemeris_df["observatory_code"].isin([observatory_code])][
                "mjd_utc"
            ].unique(),
            scale="utc",
            format="mjd",
        )
    ephemeris = ephemeris_df[["RA", "DEC"]].values

    # Use PYOORB to generate ephemeris for each target observed by
    # each observer
    ephemeris_pyoorb = generateEphemeris(
        orbits,
        observers,
        backend="PYOORB",
    )
    ephemeris_pyoorb = ephemeris_pyoorb[["RA_deg", "Dec_deg"]].values

    # pyoorb's ephemerides agree with Horizons' ephemerides
    # to within the tolerance below.
    testEphemeris(
        ephemeris_pyoorb, ephemeris, angle_tol=(10 * u.milliarcsecond), magnitude=True
    )

    return
