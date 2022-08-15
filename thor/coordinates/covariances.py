import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from astropy import units as u
from astropy.table import Table
from scipy.stats import multivariate_normal
from typing import (
    Callable,
    Hashable,
    List,
    Optional,
    Union
)

from .jacobian import calc_jacobian

logger = logging.getLogger(__name__)

__all__ = [
    "sigmas_to_covariance",
    "sample_covariance",
    "transform_covariances_sampling",
    "transform_covariances_jacobian",
    "sigmas_to_df",
    "sigmas_from_df",
    "covariances_to_df",
    "covariances_from_df",
    "covariances_to_table",
]

COVARIANCE_FILL_VALUE = 0.0

def sigmas_to_covariance(sigmas: Union[np.ndarray, np.ma.masked_array]) -> np.ma.masked_array:
    """
    Convert an array of sigmas into an array of covariance
    matrices (non-diagonal elements are assumed to be zero).

    Parameters
    ----------
    sigmas : {`~numpy.ndarray`, `~numpy.ma.masked_array`} (N, D)
        1-sigma uncertainty values for each coordinate dimension D.

    Returns
    -------
    covariances : `~numpy.ma.masked_array` (N, D, D)
        Covariance matrices with the squared 1-sigma values inserted along
        each N diagonal.
    """
    if isinstance(sigmas, (np.ma.masked_array)):
        sigmas_ = sigmas.filled()
    else:
        sigmas_ = sigmas

    N, D = sigmas_.shape
    if np.all(np.isnan(sigmas_)):
        covariances = np.ma.zeros((N, D, D),
            dtype=np.float64,
        )
        covariances.fill_value = COVARIANCE_FILL_VALUE
        covariances.mask = np.ma.ones((N, D, D), dtype=bool)

    else:
        I = np.identity(D, dtype=sigmas_.dtype)
        covariances = np.einsum('kj,ji->kij', sigmas_**2, I)
        covariances = np.ma.masked_array(
            covariances,
            dtype=np.float64,
            fill_value=COVARIANCE_FILL_VALUE
        )
        covariances.mask = np.where(np.isnan(covariances) | (covariances == COVARIANCE_FILL_VALUE), 1, 0)

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
        coords: Union[np.ndarray, np.ma.masked_array],
        covariances: Union[np.ndarray, np.ma.masked_array],
        func: Callable,
        num_samples: int = 100000
    ) -> np.ma.masked_array:
    """
    Transform covariance matrices by sampling the transformation function.

    Parameters
    ----------
    coords : {`~numpy.ndarray`, `~np.ma.masked_array`} (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : {`~numpy.ndarray`, `~np.ma.masked_array`} (N, D, D)
        Covariance matrices to transform via sampling.
    func : function
        A function that takes coords (N, D) as input and returns the transformed
        coordinates (N, D). See for example: `thor.coordinates.cartesian_to_spherical`
        or `thor.coordinates.cartesian_to_keplerian`.
    num_samples : int, optional
        The number of samples to draw.

    Returns
    -------
    covariances_out : `~np.ma.masked_array` (N, D, D)
        Transformed covariance matrices.

    Raises
    ------
    TypeError: If coords or covariances are not a `~numpy.ndarray` or a `~numpy.ma.masked_array`
    """
    if isinstance(coords, np.ma.masked_array):
        coords_ = deepcopy(coords.filled())
    elif isinstance(coords, np.ndarray):
        coords_ = deepcopy(coords)
    else:
        err = ("coords should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}")
        raise TypeError(err)

    if isinstance(covariances, np.ma.masked_array):
        covariances_ = deepcopy(covariances.filled())
    elif isinstance(covariances, np.ndarray):
        covariances_ = deepcopy(covariances)
    else:
        err = ("covariances should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}")
        raise TypeError(err)

    covariances_out = []
    for coord, covariance in zip(coords_, covariances_):
        samples = sample_covariance(coord, covariance, num_samples)
        samples_converted = func(samples)
        covariances_out.append(np.cov(samples_converted.T))

    covariances_out = np.stack(covariances_out)
    covariances_out = np.ma.masked_array(
        covariances,
        fill_value = COVARIANCE_FILL_VALUE,
        mask = np.isnan(covariances)
    )
    return covariances_out

def transform_covariances_jacobian(
        coords: Union[np.ndarray, np.ma.masked_array],
        covariances: Union[np.ndarray, np.ma.masked_array],
        _func: Callable,
        in_axes: Optional[Hashable] = (0,),
        out_axes: Optional[int] = 0,
        **kwargs,
    ) -> np.ma.masked_array:
    """
    Transform covariance matrices by calculating the Jacobian of the transformation function
    using `~jax.jacfwd`.

    Parameters
    ----------
    coords : {`~numpy.ndarray`, `~np.ma.masked_array`} (N, D)
        Coordinates that correspond to the input covariance matrices.
    covariances : {`~numpy.ndarray`, `~np.ma.masked_array`} (N, D, D)
        Covariance matrices to transform via numerical differentiation.
    _func : function
        A function that takes a single coord (D) as input and return the transformed
        coordinate (D). See for example: `thor.coordinates._cartesian_to_spherical`
        or `thor.coordinates._cartesian_to_keplerian`.
    in_axes : Optional[Hashable]
        An integer or ``None`` indicates which array axis to map over for all arguments (with ``None``
        indicating not to map any axis), and a tuple indicates which axis to map
        for each corresponding positional argument. Axis integers must be in the
        range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
        dimensions (axes) of the corresponding input array.

        From: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap
    out_axes : Optional[int]
        An integer, None, or (nested) standard Python container (tuple/list/dict) thereof
        indicating where the mapped axis should appear in the output. All outputs with a
        mapped axis must have a non-None out_axes specification.

        From: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap

    Returns
    -------
    covariances_out : `~np.ma.masked_array` (N, D, D)
        Transformed covariance matrices.

    Raises
    ------
    TypeError: If coords or covariances are not a `~numpy.ndarray` or a `~numpy.ma.masked_array`
    """
    if isinstance(coords, np.ma.masked_array):
        coords_ = deepcopy(coords.filled())
    elif isinstance(coords, np.ndarray):
        coords_ = deepcopy(coords)
    else:
        err = ("coords should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}")
        raise TypeError(err)

    if isinstance(covariances, np.ma.masked_array):
        covariances_ = deepcopy(covariances.filled())
    elif isinstance(covariances, np.ndarray):
        covariances_ = deepcopy(covariances)
    else:
        err = ("covariances should be one of {`~numpy.ndarray`, `~numpy.ma.masked_array`}")
        raise TypeError(err)

    jacobian = calc_jacobian(
        coords_,
        _func,
        in_axes=in_axes,
        out_axes=out_axes,
        **kwargs
    )

    covariances_out = jacobian @ covariances_ @ np.transpose(jacobian, axes=(0, 2, 1))
    covariances_out = np.ma.masked_array(
        covariances_out,
        fill_value=COVARIANCE_FILL_VALUE,
        mask=np.isnan(covariances_out)
    )
    return covariances_out

def sigmas_to_df(
        sigmas: np.ma.masked_array,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    ) -> pd.DataFrame:
    """
    Place sigmas into a `pandas.DataFrame`.

    Parameters
    ----------
    sigmas : `~numpy.ma.masked_array` (N, D)
        1-sigma uncertainty values for each coordinate dimension D.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    """
    N, D = sigmas.shape

    data = {}
    for i in range(D):
        data[f"sigma_{coord_names[i]}"] = sigmas[:, i]

    return pd.DataFrame(data)

def sigmas_from_df(
        df: pd.DataFrame,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
    ) -> np.ma.masked_array:
    """
    Read sigmas from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.

    Returns
    -------
    sigmas : `~numpy.ma.masked_array` (N, D)
        1-sigma uncertainty values for each coordinate dimension D.
    """
    N = len(df)
    D = len(coord_names)
    sigmas = np.ma.zeros((N, D), dtype=np.float64)
    sigmas.fill_value = COVARIANCE_FILL_VALUE
    sigmas.mask = np.ones((N, D), dtype=bool)

    for i in range(D):
        try:
            sigmas[:, i] = df[f"sigma_{coord_names[i]}"].values
            sigmas.mask[:, i] = np.where(np.isnan(sigmas[:, i]), 1, 0)

        except KeyError:
            logger.debug(f"No sigma column found for dimension {coord_names[i]}.")

    return sigmas

def covariances_to_df(
        covariances: np.ma.masked_array,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        kind: str = "lower",
    ) -> pd.DataFrame:
    """
    Place covariance matrices into a `pandas.DataFrame`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ma.masked_array` (N, D, D)
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
    ) -> np.ma.masked_array:
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
    covariances : `~numpy.ma.masked_array` (N, D, D)
        3D array of covariance matrices.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N = len(df)
    D = len(coord_names)
    covariances = np.ma.zeros(
        (N, D, D),
        dtype=np.float64
    )
    covariances.fill_value = COVARIANCE_FILL_VALUE
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
        covariances: np.ma.masked_array,
        coord_names: List[str] = ["x", "y", "z", "vx", "vy", "vz"],
        coord_units = [u.au, u.au, u.au, u.au/u.d, u.au/u.d, u.au/u.d],
        kind: str = "lower",
    ) -> Table:
    """
    Place covariance matrices into a `astropy.table.table.Table`. Splits the covariance matrices
    into either upper or lower triangular form and then adds one column per dimension.

    Parameters
    ----------
    covariances : `~numpy.ma.masked_array` (N, D, D)
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
    ) -> np.ma.masked_array:
    """
    Read covariance matrices from a `~astropy.table.table.Table`.

    Parameters
    ----------
    table : `~astropy.table.table.Table`
        Table containing covariances in either upper or lower triangular
        form.
    coord_names : List[str]
        Names of the coordinate columns, will be used to determine column names for
        each element in the triangular covariance matrix.
    kind : {'upper', 'lower'}
        The orientation of the triangular representation.

    Returns
    -------
    covariances : `~numpy.ma.masked_array` (N, D, D)
        3D array of covariance matrices.

    Raises
    ------
    ValueError : If kind is not one of {'upper', 'lower'}
    """
    N = len(table)
    D = len(coord_names)
    covariances = np.ma.zeros(
        (N, D, D),
        dtype=np.float64
    )
    covariances.fill_value = COVARIANCE_FILL_VALUE
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