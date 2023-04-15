import os

import numpy as np
import pandas as pd
from astropy.time import Time

from ...constants import Constants as c
from ..state_transition import calcStateTransitionMatrix
from ..universal_propagate import propagateUniversal

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../testing/data"
)
DT = np.array([100.0])
MU = c.MU


def test_calcStateTransitionMatrix_zerodt():

    orbit = np.array([1.0, 0.0, 0.0, 0.0002, 0.0002, 0.0])
    dt = 0.0

    phi = calcStateTransitionMatrix(orbit, dt, mu=MU, max_iter=100, tol=1e-15)

    # When dt = 0, the state transition matrix should be the
    # the identity matrix (6, 6)
    np.testing.assert_array_equal(np.identity(6), phi)
    return
