import os
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...testing import testOrbits
from ..orbits import Orbits
from ..propagate import propagateOrbits
from ..lambert import calcLambert

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)

MU = c.MU
DT = np.array([0, 5])

def test_calcLambert():

    # Read vectors from the test data set
    vectors_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )

    # Limit vectors and elements to elliptical orbits only
    elliptical_vectors = (
        (~vectors_df["orbit_class"].str.contains("Hyperbolic"))
        & (~vectors_df["orbit_class"].str.contains("Parabolic"))
    )
    vectors_df = vectors_df[elliptical_vectors]

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
    states_df = propagateOrbits(
        orbits,
        t1,
        backend="MJOLNIR",
        num_jobs=1,
        chunk_size=1
    )

    for target in targets:

        states_mask = (states_df["orbit_id"].isin([target]))
        states = states_df[states_mask]
        states = states[["x", "y", "z", "vx", "vy", "vz"]].values

        for selected_obs in [[0, 1]]:

            r0 = states[selected_obs[0], :3]
            r1 = states[selected_obs[1], :3]

            v0, v1 = calcLambert(
                r0,
                t1[selected_obs[0]].utc.mjd,
                r1,
                t1[selected_obs[1]].utc.mjd,
                mu=MU,
                max_iter=1000,
                dt_tol=1e-12
            )
            lambert_state0 = np.concatenate([r0, v0])
            lambert_state1 = np.concatenate([r1, v1])

            # Test that the resulting orbit is within the tolerances of the
            # true state below
            testOrbits(
                lambert_state0.reshape(1, -1),
                states[selected_obs[0]:selected_obs[0]+1],
                position_tol=(1e-10*u.mm),
                velocity_tol=(1*u.m/u.s),
                raise_error=False
            )
            testOrbits(
                lambert_state1.reshape(1, -1),
                states[selected_obs[1]:selected_obs[1]+1],
                position_tol=(1e-10*u.mm),
                velocity_tol=(1*u.m/u.s),
                raise_error=False
            )

    return
