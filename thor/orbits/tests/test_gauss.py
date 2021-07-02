import os
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import KERNELS_DE430
from ...utils import getSPICEKernels
from ...utils import setupSPICE
from ...testing import testOrbits
from ..gauss import gaussIOD

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)


def selectBestIOD(iod_orbits, true_orbit):
    """
    Helper function that selects the best preliminary orbit
    by selecting the orbit closes in position to the
    true orbit.

    This is intended to only used for testing.

    Parameters
    ----------
    iod_orbits : `~numpy.ndarray` (N, 6)
        Cartesian preliminary orbits from IOD functions.
    true_orbit : `~numpy.ndarray` (1, 6)
        True cartesian orbit.

    Returns
    -------
    best_iod_orbit : `~numpy.ndarray` (1, 6)
        The orbit closest in position to the true orbit.
    """
    delta_state = iod_orbits - true_orbit
    delta_position = np.linalg.norm(delta_state[:, :3], axis=1)
    nearest_iod = np.argmin(delta_position)

    return iod_orbits[nearest_iod:nearest_iod+1]

def test_gaussIOD():

    getSPICEKernels(KERNELS_DE430)
    setupSPICE(KERNELS_DE430, force=True)

    vectors_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )
    ephemeris_df = pd.read_csv(
        os.path.join(DATA_DIR, "ephemeris.csv")
    )
    observer_states_df = pd.read_csv(
        os.path.join(DATA_DIR, "observer_states.csv"),
        index_col=False
    )

    targets = ephemeris_df["targetname"].unique()
    observatories = ephemeris_df["observatory_code"].unique()

    for target in targets:
        for observatory in ["500"]:
            for selected_obs in [[23, 29, 35]]:

                ephemeris_mask = (
                    (ephemeris_df["targetname"].isin([target]))
                    & (ephemeris_df["observatory_code"].isin([observatory]))
                )
                ephemeris = ephemeris_df[ephemeris_mask]

                coords = ephemeris[["RA", "DEC"]].values
                observation_times = Time(
                    ephemeris["mjd_utc"].values,
                    format="mjd",
                    scale="utc"
                )

                vectors_mask = (vectors_df["targetname"].isin([target]))
                vectors = vectors_df[vectors_mask]
                vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values

                observer_states_mask = (observer_states_df["observatory_code"].isin([observatory]))
                observer_states = observer_states_df[observer_states_mask]
                observer_states = observer_states[["x", "y", "z", "vx", "vy", "vz"]].values

                states = ephemeris[["x", "y", "z", "vx", "vy", "vz"]].values

                # Run IOD
                iod_orbits = gaussIOD(
                    coords[selected_obs, :],
                    observation_times.utc.mjd[selected_obs],
                    observer_states[selected_obs, :3],
                    velocity_method="gibbs",
                    light_time=True,
                    max_iter=100,
                    iterate=False
                )

                # Select the best IOD orbit
                best_iod_orbit = selectBestIOD(
                    iod_orbits.cartesian,
                    states[selected_obs[1]:selected_obs[1] + 1]
                )

                # Test that the resulting orbit is within the tolerances of the
                # true state below
                testOrbits(
                    best_iod_orbit,
                    states[selected_obs[1]:selected_obs[1] + 1],
                    position_tol=(1000*u.km),
                    velocity_tol=(1*u.mm/u.s),
                    raise_error=False
                )
    return