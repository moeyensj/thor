import numpy as np

from ..covariances import sigmas_to_covariance

def test_sigmas_to_covariance():
    # Create several random arrays of 1-sigma uncertainties with
    # shape (N, D) and manually convert those into covariance matrices
    # with shape (N, D, D) (non-diagonal terms will be zeros)
    # Assert that these desired covariances are the same as the ones
    # returned by sigmas_to_covariance
    for D in np.arange(1, 6):

        sigmas1 = 1e-8 * np.random.rand(D)
        sigmas2 = 1e-6 * np.random.rand(D)
        sigmas = np.stack([sigmas1, sigmas2], axis=0)
        covariance1 = np.diag(sigmas1**2)
        covariance2 = np.diag(sigmas2**2)
        covariances_desired = np.stack([covariance1, covariance2], axis=0)

        covariances_actual = sigmas_to_covariance(sigmas).filled()

        np.testing.assert_equal(covariances_actual, covariances_desired)