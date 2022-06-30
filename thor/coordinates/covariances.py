import logging
import numpy as np
import pandas as pd
from jax import (
    config,
    jacfwd
)
import jax.numpy as jnp
from astropy import units as u
from astropy.table import Table
from scipy.stats import multivariate_normal
from typing import (
    Callable,
    List
)

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

__all__ = [
    "sigmas_to_covariance",
    "sample_covariance",
    "transform_covariances_sampling",
    "transform_covariances_jacobian",
    "covariances_to_df",
    "covariances_from_df",
    "covariances_to_table",
]

logger = logging.getLogger(__name__)

def sigmas_to_covariance(sigmas: Union[np.ndarray, np.ma.core.MaskedArray]) -> np.ma.core.MaskedArray:
    """
    Convert an array of sigmas into an array of covariance
    matrices (non-diagonal elements are assumed to be zero).

    Parameters
    ----------
    sigmas : {`~numpy.ndarray`, `~numpy.ma.core.MaskedArray`} (N, D)
        1-sigma uncertainty values for each coordinate dimension D.

    Returns
    -------
    covariances : `~numpy.ma.core.MaskedArray` (N, D, D)
        Covariance matrices with the squared 1-sigma values inserted along
        each N diagonal.
    """
    if isinstance(sigmas, (np.ma.core.MaskedArray)):
        sigmas_ = sigmas.filled()
    else:
        sigmas_ = sigmas

    N, D = sigmas_.shape
    covariances = np.ma.zeros((N, D, D), dtype=np.float64)
    covariances.fill_value = np.NaN
    covariances.mask = np.ones((N, D, D), dtype=bool)

    I = np.identity(D, dtype=sigmas_.dtype)
    covariances[:, :, :] = np.einsum('kj,ji->kij', sigmas_**2, I)
    covariances.mask = np.where(np.isnan(covariances) | (covariances == 0), 1, 0)

    return covariances

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
        **kwargs,
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
    jacobian_func = jacfwd(_func, argnums=0)

    covariances_out = []

    for i, (coord, covariance) in enumerate(zip(coords, covariances)):

        kwargs_i = {}
        for k, v in kwargs.items():
            if isinstance(v, (list, np.ndarray, jnp.ndarray)):
                kwargs_i[k] = v[i]
            else:
                kwargs_i[k] = v

        jacobian = jacobian_func(coord, **kwargs_i)
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

def covariances_to_table(
        covariances: np.ma.MaskedArray,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        coord_units = [u.au, u.au, u.au, u.au/u.d, u.au/u.d, u.au/u.d],
        kind: str = "lower",
    ) -> Table:
    """
    Place covariance matrices into a `astropy.table.table.Table`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ma.core.MaskedArray` (N, D, D)
        3D array of covariance matrices.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    coord_units : List[]
        The unit for each coordinate, will be used to determination the units for
        element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    table : `~astropy.table.table.Table`
        Table containing covariances in either upper or lower triangular
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
        data[f"cov_{coord_names[i]}_{coord_names[j]}"] = covariances[:, i, j] * coord_units[i] * coord_units[j]

    return Table(data)

def covariances_from_table(
        table: Table,
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
    N = len(table)
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
            covariances[:, i, j] = table[f"cov_{coord_names[i]}_{coord_names[j]}"].values
            covariances[:, j, i] = covariances[:, i, j]
        except KeyError:
            logger.debug(f"No covariance column found for dimensions {coord_names[i]},{coord_names[j]}.")

    return covariances