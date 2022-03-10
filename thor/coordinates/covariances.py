import numpy as np
from jax import config, jacfwd
from scipy.stats import multivariate_normal
from typing import Callable

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

__all__ = [
    "sample_covariance",
    "transform_covariances_sampling",
    "transform_covariances_jacobian",
]

def sample_covariance(
        mean: np.ndarray,
        cov: np.ndarray,
        num_samples: int = 100000
    ) -> np.ndarray:
    """
    Sample a multivariate Gaussian distribution with given
    mean and covariances.

    Parameters
    ----------
    mean : `~numpy.ndarray` (D)
        Multivariate mean of the Gaussian distribution.
    cov : `~numpy.ndarray` (D, D)
        Multivariate variance-covariance matrix of the Gaussian distribution.
    num_samples : int, optional
        Number of samples to draw from the distribution.

    Returns
    -------
    samples : `~numpy.ndarray` (num_samples, D)
        The drawn samples row-by-row.
    """
    normal = multivariate_normal(
        mean=mean,
        cov=cov,
        allow_singular=True
    )
    samples = normal.rvs(num_samples)
    return samples

def transform_covariances_sampling(
        coords: np.ndarray,
        covariances: np.ndarray,
        func: Callable,
        num_samples: int = 100000
    ) -> np.ndarray:
    """
    Transform covariance matrices by sampling the transformation function.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices to transform via sampling.
    func : function
        A function that takes coords (N, D) as input and returns the transformed
        coordinates (N, D). See for example: `thor.coordinates.cartesian_to_spherical`
        or `thor.coordinates.cartesian_to_keplerian`.
    num_samples : int, optional
        The number of samples to draw.

    Returns
    -------
    covariances_out : `~numpy.ndarray` (N, D, D)
        Transformed covariance matrices.
    """
    covariances_out = []
    for coord, covariance in zip(coords, covariances):
        samples = sample_covariance(coord, covariance, num_samples)
        samples_converted = func(samples)
        covariances_out.append(np.cov(samples_converted.T))

    return np.stack(covariances_out)

def transform_covariances_jacobian(
        coords: np.ndarray,
        covariances: np.ndarray,
        _func: Callable,
    ) -> np.ndarray:
    """
    Transform covariance matrices by calculating the Jacobian of the transformation function
    using `~jax.jacfwd`.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : `~numpy.ndarray` (N, D, D)
        Covariance matrices to transform via numerical differentiation.
    _func : function
        A function that takes a single coord (D) as input and return the transformed
        coordinate (D). See for example: `thor.coordinates._cartesian_to_spherical`
        or `thor.coordinates._cartesian_to_keplerian`.

    Returns
    -------
    covariances_out : `~numpy.ndarray` (N, D, D)
        Transformed covariance matrices.
    """
    # Calculate the jacobian function for the input function
    # Do this only once!
    jacobian_func = jacfwd(_func)

    covariances_out = []
    for coord, covariance in zip(coords, covariances):
        jacobian = jacobian_func(coord)
        covariance_out = np.array(jacobian @ covariance @ jacobian.T)
        covariances_out.append(covariance_out)

    return np.stack(covariances_out)
