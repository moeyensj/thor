import pytest
import numpy as np
import numpy.testing as npt

from ...constants import Constants as c
from ..state_transition import calc_state_transition_matrix

### Tests last updated: 2022-08-25

MU = c.MU

def test_calcStateTransitionMatrix_zerodt():

    orbit = np.array([1., 0., 0., 0.0002, 0.0002, 0.])
    dt = 0.0

    phi = calc_state_transition_matrix(
        orbit,
        dt,
        mu=MU,
        max_iter=100,
        tol=1e-15
    )

    # When dt = 0, the state transition matrix should be the
    # the identity matrix (6, 6)
    npt.assert_array_equal(np.identity(6), phi)
    return

