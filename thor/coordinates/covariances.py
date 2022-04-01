import logging
import numpy as np
import pandas as pd
from jax import (
    config,
    jacfwd
)
from scipy.stats import multivariate_normal
from typing import (
    Callable,
    List
)

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

__all__ = [
    "sample_covariance",
    "transform_covariances_sampling",
    "transform_covariances_jacobian",
    "covariances_to_df",
    "covariances_from_df"
]

logger = logging.getLogger(__file__)

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

def covariances_to_df(
        covariances: np.ma.MaskedArray,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        kind: str = "lower",
    ) -> pd.DataFrame:
    """
    Place covariance matrices into a `pandas.DataFrame`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ma.core.MaskedArray` (N, D, D)
        3D array of covariance matrices.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}

    """
    N, D, D = covariances.shape

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = (
            "kind should be one of {'upper', 'lower'}"
        )
        raise ValueError(err)

    data = {}
    for i, j in zip(ii, jj):
        data[f"cov_{coord_names[i]}_{coord_names[j]}"] = covariances[:, i, j]

    return pd.DataFrame(data)

def covariances_from_df(
        df: pd.DataFrame,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        kind: str = "lower"
    ) -> np.ma.MaskedArray:
    """
    Read covariance matrices from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    covariances : `~numpy.ma.core.MaskedArray` (N, D, D)
        3D array of covariance matrices.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N = len(df)
    D = len(coord_names)
    covariances = np.ma.zeros((N, D, D), dtype=np.float64)
    covariances.fill_value = np.NaN
    covariances.mask = np.ones((N, D, D), dtype=bool)

    if kind == "upper":
        ii, jj = np.triu_indices(D)
    elif kind == "lower":
        ii, jj = np.tril_indices(D)
    else:
        err = (
            "kind should be one of {'upper', 'lower'}"
        )
        raise ValueError(err)

    for i, j in zip(ii, jj):
        try:
            covariances[:, i, j] = df[f"cov_{coord_names[i]}_{coord_names[j]}"].values
            covariances[:, j, i] = covariances[:, i, j]
        except KeyError:
            logger.debug(f"No covariance column found for dimensions {coord_names[i]},{coord_names[j]}.")

    return covariances
