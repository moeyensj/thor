import os

import pandas as pd
import pytest
from astropy import units as u
from astropy.time import Time

from thor.utils.spice import getSPICEKernels

from ...testing import testOrbits
from ...utils import KERNELS_DE440, getMPCObservatoryCodes, getSPICEKernels, setupSPICE
from ..state import getObserverState

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../testing/data"
)


def test_getObserverState_heliocentric():
    """
    Read the test dataset for heliocentric state vectors of each observatory at each observation time.
    Use THOR to find heliocentric state vectors of each observatory at each observation time.
    Compare the resulting state vectors and test how well they agree with the ones pulled from Horizons.
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    # Make sure the latest version of the MPC observatory codes
    # has been downloaded
    getMPCObservatoryCodes()

    # Read observatory states from test data file
    observer_states_df = pd.read_csv(
        os.path.join(DATA_DIR, "observer_states.csv"), index_col=False
    )

    origin = "heliocenter"
    for observatory_code in observer_states_df["observatory_code"].unique():
        observatory_mask = observer_states_df["observatory_code"].isin(
            [observatory_code]
        )

        times = Time(
            observer_states_df[observatory_mask]["mjd_utc"].values,
            scale="utc",
            format="mjd",
        )
        observer_states = observer_states_df[observatory_mask][
            ["x", "y", "z", "vx", "vy", "vz"]
        ].values

        observer_states_thor = getObserverState(
            [observatory_code],
            times,
            origin=origin,
        )
        observer_states_thor = observer_states_thor[
            ["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]
        ].values

        # Test that each state agrees with Horizons
        # to within the tolerances below
        testOrbits(
            observer_states_thor,
            observer_states,
            position_tol=(20 * u.m),
            velocity_tol=(1 * u.cm / u.s),
            magnitude=True,
        )
    return


def test_getObserverState_barycentric():
    """
    Read the test dataset for barycentric state vectors of each observatory at each observation time.
    Use THOR to find barycentric state vectors of each observatory at each observation time.
    Compare the resulting state vectors and test how well they agree with the ones pulled from Horizons.
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    # Make sure the latest version of the MPC observatory codes
    # has been downloaded
    getMPCObservatoryCodes()

    # Read observatory states from test data file
    observer_states_df = pd.read_csv(
        os.path.join(DATA_DIR, "observer_states_barycentric.csv"), index_col=False
    )

    origin = "barycenter"
    for observatory_code in observer_states_df["observatory_code"].unique():
        observatory_mask = observer_states_df["observatory_code"].isin(
            [observatory_code]
        )

        times = Time(
            observer_states_df[observatory_mask]["mjd_utc"].values,
            scale="utc",
            format="mjd",
        )
        observer_states = observer_states_df[observatory_mask][
            ["x", "y", "z", "vx", "vy", "vz"]
        ].values

        observer_states_thor = getObserverState(
            [observatory_code],
            times,
            origin=origin,
        )
        observer_states_thor = observer_states_thor[
            ["obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz"]
        ].values

        # Test that each state agrees with Horizons
        # to within the tolerances below
        testOrbits(
            observer_states_thor,
            observer_states,
            position_tol=(20 * u.m),
            velocity_tol=(1 * u.cm / u.s),
            magnitude=True,
        )
    return


def test_getObserverState_raises():

    times = Time([59000], scale="utc", format="mjd")

    with pytest.raises(ValueError):
        # Raise error for incorrect frame
        observer_states = getObserverState(["500"], times, frame="eccliptic")

        # Raise error for incorrect origin
        observer_states = getObserverState(["500"], times, origin="heeliocenter")

    with pytest.raises(TypeError):
        # Raise error for non-astropy time
        observer_states = getObserverState(["500"], times.tdb.mjd)
