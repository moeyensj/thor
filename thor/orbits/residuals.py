import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

__all__ = [
    "calcResiduals",
    "calcSimpleResiduals",
    "calcProbabilisticResiduals"
]

def calcResiduals(
        coords_actual,
        coords_desired,
        sigmas_actual=None,
        covariances_actual=None,
        include_probabilistic=True,
    ):
    """
    Calculate residuals (actual - desired) and the associated chi2 if the
    1-sigma uncertainties on the actual quantities are provided. If desired,
    a probabilistic measure of residual can also be calculated using the 1-sigma
    uncertainties (which are converted into a covariance matrix) or by providing the
    coveriance matrices for each coordinate. The probabilistic statistics include
    the chi2 for each measurement, the mahalanobis distance and the associated
    probability (calculated as 1 - chi2.cdf(mahalanobis_distance, degrees of freedom)).

    Parameters
    ----------
    coords_actual : `~numpy.ndarray` (N, M)
        Actual N coordinates in M dimensions.
    coords_desired : `~numpy.ndarray` (N, M)
        The desired N coordinates in M dimensions.
    sigmas_actual : `~numpy.ndarray` (N, M), optional
        The 1-sigma uncertainties of the actual coordinates. Can be
        None, in which case chi2 will be return as a NaN.
    covariances_actual : list of N `~numpy.ndarray`s (M, M)
        The covariance matrix in M dimensions for each
        actual observation if available.
    include_probabilistic : bool, optional
        Include the calculation of mahalanobis distance and the associated
        probability.

    Returns
    -------
    residuals : `~numpy.ndarray` (N, 2)
        Residuals for each coordinate (actual - desired)
    stats : tuple (1 or 3)
        chi2 :  `~numpy.ndarray` (N)
            Chi-squared for each observation given the 1-sigma
            uncertainties. NaN if no sigmas_actual is None.
        p : `~numpy.ndarray` (N)
            The probability that the actual coordinates given their uncertainty
            belong to the same multivariate normal distribution as the desired
            coordinates.
        d : `~numpy.ndarray` (N)
            The Mahalanobis distance of each coordinate compared to the desired
            coordinates.
    """
    if covariances_actual is None and sigmas_actual is not None and include_probabilistic:
        covariances_actual_ = [np.diag(i**2) for i in sigmas_actual]
        sigmas_actual_ = sigmas_actual
    elif covariances_actual is not None and sigmas_actual is None:
        sigmas_actual_ = np.zeros_like(coords_actual)
        for i in range(len(coords_actual)):
            sigmas_actual_[i] = np.diagonal(covariances_actual[i])
        sigmas_actual_ = np.sqrt(sigmas_actual_)
    else:
        covariances_actual_ = covariances_actual
        sigmas_actual_ = sigmas_actual

    residuals, chi2 = calcSimpleResiduals(
        coords_actual,
        coords_desired,
        sigmas_actual=sigmas_actual_
    )

    if include_probabilistic:
        p, d = calcProbabilisticResiduals(
            coords_actual,
            coords_desired,
            covariances_actual_
        )
        stats = (chi2, p, d)
    else:

        stats = (chi2,)

    return residuals, stats


def calcSimpleResiduals(
        coords_actual,
        coords_desired,
        sigmas_actual=None,
    ):
    """
    Calculate residuals and the associated chi2 if the
    1-sigma uncertainties on the actual quantities are provided.

    Parameters
    ----------
    coords_actual : `~numpy.ndarray` (N, M)
        Actual N coordinates in M dimensions.
    coords_desired : `~numpy.ndarray` (N, M)
        The desired N coordinates in M dimensions.
    sigmas_actual : `~numpy.ndarray` (N, M), optional
        The 1-sigma uncertainties of the actual coordinates. Can be
        None, in which case chi2 will be return as a NaN.

    Returns
    -------
    residuals : `~numpy.ndarray` (N, 2)
        Residuals for each coordinate (actual - desired)
    chi2 : `~numpy.ndarray` (N)
        Chi-squared for each observation given the 1-sigma
        uncertainties. NaN if no sigmas_actual is None.
    """
    residuals = np.zeros_like(coords_actual)

    ra = coords_actual[:, 0]
    dec = coords_actual[:, 1]
    ra_pred = coords_desired[:, 0]
    dec_pred = coords_desired[:, 1]

    # Calculate residuals in RA, make sure to fix any potential wrap around errors
    residual_ra = (ra - ra_pred)
    residual_ra = np.where(residual_ra > 180., 360. - residual_ra, residual_ra)
    residual_ra *= np.cos(np.radians(dec_pred))

    # Calculate residuals in Dec
    residual_dec = dec - dec_pred

    if isinstance(sigmas_actual, np.ndarray):
        sigma_ra = sigmas_actual[:, 0]
        sigma_dec = sigmas_actual[:, 1]

        # Calculate chi2
        chi2 = ((residual_ra**2 / sigma_ra**2)
            + (residual_dec**2 / sigma_dec**2))
    else:
        chi2 = np.NaN

    residuals[:, 0] = residual_ra
    residuals[:, 1] = residual_dec

    return residuals, chi2

def calcProbabilisticResiduals(
        coords_actual,
        coords_desired,
        covariances_actual
    ):
    """
    Calculate the probabilistic residual.

    Parameters
    ----------
    coords_actual : `~numpy.ndarray` (N, M)
        Actual N coordinates in M dimensions.
    coords_desired : `~numpy.ndarray` (N, M)
        The desired N coordinates in M dimensions.
    sigmas_actual : `~numpy.ndarray` (N, M)
        The 1-sigma uncertainties of the actual coordinates.
    covariances_actual : list of N `~numpy.ndarray`s (M, M)
        The covariance matrix in M dimensions for each
        actual observation if available.

    Returns
    -------
    p : `~numpy.ndarray` (N)
        The probability that the actual coordinates given their uncertainty
        belong to the same multivariate normal distribution as the desired
        coordinates.
    d : `~numpy.ndarray` (N)
        The Mahalanobis distance of each coordinate compared to the desired
        coordinates.
    """
    d = np.zeros(len(coords_actual))
    p = np.zeros(len(coords_actual))

    for i, (actual, desired, covar) in enumerate(zip(coords_actual, coords_desired, covariances_actual)):
        # Calculate the degrees of freedom
        k = len(actual)

        # Calculate the mahalanobis distance between the two coordinates
        d_i = mahalanobis(
            actual,
            desired,
            np.linalg.inv(covar)
        )

        # Calculate the probability that both sets of coordinates are drawn from
        # the same multivariate normal
        p_i = 1 - chi2.cdf(d_i, k)

        # Append results
        d[i] = d_i
        p[i] = p_i

    return p, d
