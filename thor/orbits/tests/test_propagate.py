import os
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u

from ...testing import testOrbits
from ..orbits import Orbits
from ..propagate import propagateOrbits

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)
DT = np.arange(-1000, 1000, 5)

def test_propagateOrbits():
    """
    Read the test dataset for the initial state vectors of each target at t0, then propagate
    those states to all t1 using THOR's 2-body propagator and OORB's 2-body propagator (via pyoorb).
    Compare the resulting states and test how well they agree.
    """
    # Read vectors and elements from test data set
    vectors_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )

    # Get the target names
    targets = vectors_df["targetname"].unique()

    # Get the initial epochs
    t0 = Time(
        vectors_df["mjd_tdb"].values,
        scale="tdb",
        format="mjd"
    )

    # Set propagation epochs
    t1 = t0[0] + DT

    # Pull state vectors
    vectors = vectors_df[["x", "y", "z", "vx", "vy", "vz"]].values

    # Create orbits class
    orbits = Orbits(
        vectors,
        t0,
        ids=targets
    )

    # Propagate the state at T0 to all T1 using MJOLNIR 2-body
    states_mjolnir = propagateOrbits(
        orbits,
        t1,
        backend="MJOLNIR",
        backend_kwargs={},
        num_jobs=1,
        chunk_size=1
    )
    states_mjolnir = states_mjolnir[["x", "y", "z", "vx", "vy", "vz"]].values

    # Propagate the state at T0 to all T1 using PYOORB 2-body
    states_pyoorb = propagateOrbits(
        orbits,
        t1,
        backend="PYOORB",
        backend_kwargs={"dynamical_model" : "2"},
        num_jobs=1,
        chunk_size=1
    )
    states_pyoorb = states_pyoorb[["x", "y", "z", "vx", "vy", "vz"]].values

    # Test that the propagated states agree to within the tolerances below
    testOrbits(
       states_mjolnir,
       states_pyoorb,
       orbit_type="cartesian",
       position_tol=200*u.m,
       velocity_tol=(1*u.cm/u.s),
       magnitude=True
    )
    return
