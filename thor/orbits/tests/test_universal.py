import os
import numpy as np
import pandas as pd
import spiceypy as sp
from astropy.time import Time
from astropy import units as u

from ...constants import Constants as c
from ...utils import KERNELS_DE430
from ...utils import getSPICEKernels
from ...utils import setupSPICE
from ...testing import testOrbits
from ..universal_propagate import propagateUniversal

MU = c.MU
DT = np.arange(-1000, 1000, 5)
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../testing/data"
)

def test_propagateUniversal():
    """
    Read the test dataset for the initial state vectors of each target at t1, then propagate
    those states to all t1 using THOR's 2-body propagator and SPICE's 2-body propagator (via spiceypy).
    Compare the resulting states and test how well they agree.
    """
    getSPICEKernels(KERNELS_DE430)
    setupSPICE(KERNELS_DE430, force=True)

    # Read vectors from test data set
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
    
    # Propagate initial states to each T1 using SPICE
    states_spice = []
    for i, target in enumerate(targets): 
        for dt in DT:
            states_spice.append(sp.prop2b(MU, list(vectors[i, :]), dt))
    states_spice = np.array(states_spice)
            
    # Repeat but now using THOR's universal 2-body propagator
    states_thor = propagateUniversal(
        vectors, 
        t0.tdb.mjd, 
        t1.tdb.mjd,  
        mu=MU, 
        max_iter=1000, 
        tol=1e-15
    )

    # Test 2-body propagation using THOR is
    # is within this tolerance of SPICE 2-body
    # propagation
    testOrbits(
       states_thor[:, 2:], 
       states_spice,
       orbit_type="cartesian", 
       position_tol=1*u.cm, 
       velocity_tol=(1*u.mm/u.s), 
       magnitude=True
    )
    return