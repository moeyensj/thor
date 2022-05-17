import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from typing import (
    List,
    Optional,
    Union
)
from collections import OrderedDict

from ..utils import (
    Indexable,
    times_to_df,
    times_from_df
)
from .covariances import (
    covariances_from_df,
    covariances_to_df,
)

__all__ = [
    "_ingest_coordinate",
    "_ingest_covariance",
    "Coordinates",
]

def _ingest_coordinate(
        q: Union[list, np.ndarray],
        d: int,
        coords: Optional[np.ma.core.MaskedArray] = None,
        D: int = 6,
    ) -> np.ma.core.MaskedArray:
    """
    Ingest coordinates along an axis (like the x, y, z) and add them to an existing masked array
    of coordinate measurements if that object already exists. If that object doesn't exist then
    create it and return it. Any missing values in q should be represented with NaNs.

    Parameters
    ----------
    q : list or `~numpy.ndarray` (N)
        List or 1-D array of coordinate measurements.
    d : int
        The coordinate axis (as an index). For example, for a 6D Cartesian
        state vector, the x-axis takes the 0th index, the y-axis takes the 1st index,
        the z axis takes the 2nd index, the x-velocity takes the 3rd index, etc..
    coords : `~numpy.ma.ndarray` (N, D), optional
        If coordinates (ie, other axes) have already been defined then pass them here
        so that current axis of coordinates can be added.
    D : int, optional
        Total number of dimensions represented by the coordinates.

    Returns
    -------
    coords : `~numpy.ma.array` (N, D)
        Masked array of 6D coordinate measurements with q measurements ingested.

    Raises
    ------
    ValueError
        If the length of q doesn't match the length of coords.
    """
    if q is not None:
        q_ = np.asarray(q)
        N_ = len(q_)
        if coords is None:
            coords = np.ma.zeros((N_, D), dtype=np.float64, fill_value=np.NaN)
            coords.mask = 1
        else:
            N, D = coords.shape
            if N != N_:
                err = (
                    "q needs to be the same length as the existing coordinates.\n"
                    f"q has length {N_} while coords has {N} coordinates in 6 dimensions."
                )
                raise ValueError(err)

        coords[:, d] = q_
        coords.mask[:, d] = np.where(np.isnan(q_), 1, 0)

    return coords

def _ingest_covariance(
        coords: np.ma.core.MaskedArray,
        covariance: Union[np.ndarray, np.ma.core.MaskedArray],
    ) -> np.ma.core.MaskedArray:
    """
    Ingest a set of covariance matrices.

    Parameters
    ----------
    coords : `~numpy.ma.array` (N, D)
        Masked array of 6D coordinate measurements with q measurements ingested.
    covariance : `~numpy.ndarray` or `~numpy.ma.array` (N, <=6, <=6)
        Covariance matrices for each coordinate. These matrices may have fewer dimensions
        than 6. If so, additional dimensions will be added for each masked or missing coordinate
        dimension.

    Returns
    -------
    covariance : `~numpy.ma.array` (N, D, D)
        Masked array of covariance matrices.

    Raises
    ------
    ValueError
        If not every coordinate has an associated covariance.
        If the number of covariance dimensions does not match
            the number of unmasked or missing coordinate dimensions.
    """
    D = coords.shape[1]
    axes = D - np.sum(coords.mask.all(axis=0))

    if covariance.shape[0] != len(coords):
        err = (
            "Every coordinate in coords should have an associated covariance."
        )
        raise ValueError(err)

    if covariance.shape[1] != covariance.shape[2] != axes:
        err = (
            f"Coordinates have {axes} defined dimensions, expected covariance matrix\n",
            f"shapes of (N, {axes}, {axes}."
        )
        raise ValueError(err)

    if isinstance(covariance, np.ma.core.MaskedArray) and (covariance.shape[1] == covariance.shape[2] == coords.shape[1]):
        return covariance

    covariance_ = np.ma.zeros((len(coords), D, D), dtype=np.float64, fill_value=np.NaN)
    covariance_.mask = np.zeros_like(covariance_, dtype=bool)

    for n in range(len(coords)):
        covariance_[n].mask[coords[n].mask, :] = 1
        covariance_[n].mask[:, coords[n].mask] = 1
        covariance_[n][~covariance_[n].mask] = covariance[n][~covariance_[n].mask].flatten()

    return covariance_

class Coordinates(Indexable):

    def __init__(
            self,
            *args,
            covariances: Optional[Union[np.ndarray, np.ma.array, List]] = None,
            times: Optional[Time] = None,
            origin: Optional[Union[np.ndarray, str]] = "heliocenter",
            frame: str = "ecliptic",
            names: OrderedDict = OrderedDict(),
            units: OrderedDict = OrderedDict(),
        ):
        coords = None

        # Total number of coordinate dimensions
        D = len(args)
        for d, q in enumerate(args):
            coords = _ingest_coordinate(q, d, coords, D=D)

        self._times = times
        self._values = coords
        if origin is not None:
            if isinstance(origin, str):
                self._origin = np.empty(len(self), dtype="<U16")
                self._origin.fill(origin)
            elif isinstance(origin, np.ndarray):
                assert len(origin) == len(self._values)
                self._origin = origin
            else:
                err = (
                    "Origin should be a str or `~numpy.ndarray`"
                )
                raise TypeError(err)
        else:
            self._origin = origin

        self._frame = frame
        self._names = names
        self._units = units

        if covariances is not None:
            self._covariances = _ingest_covariance(coords, covariances)
        else:
            self._covariances = None

        return

    def __len__(self):
        return len(self.values)

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values

    @property
    def covariances(self):
        return self._covariances

    @property
    def sigmas(self):

        sigmas = None
        if self._covariances is not None:
            cov_diag = np.diagonal(self._covariances, axis1=1, axis2=2)
            sigmas = np.sqrt(cov_diag)

        return sigmas

    @property
    def origin(self):
        return self._origin

    @property
    def frame(self):
        return self._frame

    @property
    def names(self):
        return self._names

    @property
    def units(self):
        return self._units

    def to_cartesian(self):
        raise NotImplementedError

    def from_cartesian(cls, cartesian):
        raise NotImplementedError

    def to_df(self,
            time_scale="utc",
        ):
        data = {}
        N, D = self.values.shape

        if self.times is not None:
            df = times_to_df(self.times, time_scale=time_scale)
        else:
            df = pd.DataFrame(index=np.arange(0, len(self)))

        for i, (k, v) in enumerate(self.names.items()):
            data[k] = self.values.filled()[:, i]

        df = df.join(pd.DataFrame(data))
        if self.covariances is not None:
            df_covariances = covariances_to_df(
                self.covariances,
                list(self.names.keys()),
                kind="lower"
            )
            df = df.join(df_covariances)

        df.insert(len(df.columns), "origin", self.origin)
        return df

    @staticmethod
    def _dict_from_df(
            df,
            coord_cols=OrderedDict(),
            origin_col="origin"
        ):
        """
        Create a dictionary from a dataframe.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing coordinates and optionally their
            times and covariances.
        coord_cols : OrderedDict
            Ordered dictionary containing as keys the coordinate dimensions and their equivalent columns
            as values. For example,
                coord_cols = OrderedDict()
                coord_cols["a"] = Column name of semi-major axis values
                coord_cols["e"] = Column name of eccentricity values
                coord_cols["i"] = Column name of inclination values
                coord_cols["raan"] = Column name of longitude of ascending node values
                coord_cols["ap"] = Column name of argument of pericenter values
                coord_cols["M"] = Column name of mean anomaly values
        origin_col : str
            Name of the column containing the origin of each coordinate.
        """
        data = {}
        data["times"] = times_from_df(df)
        for i, (k, v) in enumerate(coord_cols.items()):
            if v in df.columns:
                data[k] = df[v].values

        if origin_col in df.columns:
            data["origin"] = df[origin_col].values

        data["covariances"] = covariances_from_df(
            df,
            coord_names=list(coord_cols.keys()),
            kind="lower"
        )
        data["names"] = coord_cols

        return data