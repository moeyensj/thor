import numpy as np

from ..orbit import calcNhat

def test_calcNhat_normalized():
    state_vectors = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.002, 0.000],
        [1.0, 0.0, 1.0, 0.0, 0.002, 0.000],
        [1.0, 0.0, 1.0, 0.0, 0.000, 0.002]
    ])
    n_hat = calcNhat(state_vectors)
    n_hat_norm = np.linalg.norm(n_hat, axis=1)

    # Each vector should have magnitude of 1.0
    np.testing.assert_equal(np.ones(len(n_hat)), n_hat_norm)
    return

def test_calcNhat_orientation():
    state_vectors = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.002, 0.000],
        [1.0, 0.0, 1.0, 0.0, 0.002, 0.000],
        [1.0, 0.0, 1.0, 0.0, 0.000, 0.002]
    ])
    n_hat = calcNhat(state_vectors)

    # First orbit lies in the x-y plane and so should have a unit
    # vector normal equivalent to the z-axis
    np.testing.assert_equal(np.array([0., 0., 1.0]), n_hat[0])

    # Second orbit lies in a plane inclined 45 degrees from the x-y plane, intersecting
    # the x-y plane along the y-axis.
    np.testing.assert_equal(np.array([-np.sqrt(2)/2, 0.0, np.sqrt(2)/2]), n_hat[1])

    # Third orbit lies in a plane inclined 90 degrees from the x-y plane, intersecting
    # the x-y plane along the x-axis.
    np.testing.assert_equal(np.array([0.0, -1.0, 0]), n_hat[2])

    return