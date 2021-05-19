import os
import pytest
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from ...utils import KERNELS_DE440
from ...utils import setupSPICE
from ...utils import getSPICEKernels
from ...testing import testOrbits
from ..state import shiftOrbitsOrigin

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)

def test_shiftOrbitsOrigin():
    """
    Read test data set for initial for initial state vectors of each target at t0 in the heliocentric
    and barycentric frames, use THOR to shift each vector to a different origin and compare. 
    """
    getSPICEKernels(KERNELS_DE440)
    setupSPICE(KERNELS_DE440, force=True)

    # Read vectors from test data set
    vectors_heliocentric_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors.csv")
    )
    vectors_barycentric_df = pd.read_csv(
        os.path.join(DATA_DIR, "vectors_barycentric.csv")
    )
    vectors_heliocentric = vectors_heliocentric_df[["x", "y", "z", "vx", "vy", "vz"]].values
    vectors_barycentric = vectors_barycentric_df[["x", "y", "z", "vx", "vy", "vz"]].values

    # Get the initial epochs
    t0 = Time(
        vectors_heliocentric_df["mjd_tdb"].values,
        scale="tdb",
        format="mjd"
    )

    # Shift origin of heliocentric vectors to barycenter
    thor_barycentric_vectors = shiftOrbitsOrigin(
        vectors_heliocentric, 
        t0, 
        origin_in="heliocenter",
        origin_out="barycenter"
    )

    # Test that THOR barycentric states agree with
    # Horizons' barycentric states to within the 
    # tolerances below
    testOrbits(
        thor_barycentric_vectors,
        vectors_barycentric,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    # Shift origin of heliocentric vectors to barycenter
    thor_heliocentric_vectors = shiftOrbitsOrigin(
        vectors_barycentric, 
        t0, 
        origin_in="barycenter",
        origin_out="heliocenter"
    )

    # Test that THOR heliocentric states agree with
    # Horizons' heliocentric states to within the 
    # tolerances below
    testOrbits(
        thor_heliocentric_vectors,
        vectors_heliocentric,
        orbit_type="cartesian",
        position_tol=(1*u.cm),
        velocity_tol=(1*u.mm/u.s),
        magnitude=True
    )

    return

def test_shiftOrbitsOrigin_raise():

    with pytest.raises(ValueError):

        t1 = Time(
            np.arange(54000, 64000, 1),
            scale="tdb",
            format="mjd"
        )

        # Raise error for incorrect origin_in
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(t1), 6), dtype=float), 
            t1, 
            origin_in="baarycenter",
            origin_out="heliocenter")

        # Raise error for incorrect origin_out
        thor_helio_to_bary = shiftOrbitsOrigin(
            np.zeros((len(t1), 6), dtype=float), 
            t1, 
            origin_in="barycenter",
            origin_out="heeliocenter")

    return
