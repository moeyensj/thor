import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from copy import deepcopy
from typing import (
    List,
    Optional,
    Union
)

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
CARTESIAN_UNITS = [u.au, u.au, u.au, u.au / u.d, u.au / u.d, u.au / u.d]

SPHERICAL_COLS = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
SPHERICAL_UNITS = [u.au, u.degree, u.degree, u.au / u.d, u.degree / u.d, u.degree / u.d]

__all__ = [
    "_ingest_coordinate",
    "_ingest_covariance",
    "Coordinates",
    "CartesianCoordinates",
    "SphericalCoordinates"
]

def _ingest_coordinate(
        q: Union[list, np.ndarray],
        d: int,
        coords: Optional[np.ma.core.MaskedArray] = None
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

    Returns
    -------
    coords : `~numpy.ma.array` (N, 6)
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
            coords = np.ma.zeros((N_, 6), fill_value=np.NaN)
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
    coords : `~numpy.ma.array` (N, 6)
        Masked array of 6D coordinate measurements with q measurements ingested.
    covariance : `~numpy.ndarray` or `~numpy.ma.array` (N, <=6, <=6)
        Covariance matrices for each coordinate. These matrices may have fewer dimensions
        than 6. If so, additional dimensions will be added for each masked or missing coordinate
        dimension.

    Returns
    -------
    covariance : `~numpy.ma.array` (N, 6, 6)
        Masked array of covariance matrices.

    Raises
    ------
    ValueError
        If not every coordinate has an associated covariance.
        If the number of covariance dimensions does not match
            the number of unmasked or missing coordinate dimensions.
    """
    axes = coords.shape[1] - np.sum(coords.mask.all(axis=0))
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

    covariance_ = np.ma.zeros((len(coords), 6, 6), fill_value=np.NaN)
    covariance_.mask = np.zeros_like(covariance_, dtype=bool)

    for n in range(len(coords)):
        covariance_[n].mask[coords[n].mask, :] = 1
        covariance_[n].mask[:, coords[n].mask] = 1
        covariance_[n][~covariance_[n].mask] = covariance[n].flatten()

    return covariance_

class Coordinates:

    def __init__(
            self,
            coords: np.ma.core.MaskedArray,
            covariance: Optional[np.ma.core.MaskedArray],
            time: Optional[Time] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic",
            names: List[str] = [],
        ):
        self._values = coords
        self._time = time
        self._origin = origin
        self._frame = frame
        self._names = names

        if covariance is not None:
            self._covariance = _ingest_covariance(coords, covariance)
        else:
            self._covariance = None
        return

    def __len__(self):
        return len(self.values)

    def _handle_index(self, i):
        if isinstance(i, int):
            if i < 0:
                _i = i + len(self)
            else:
                _i = i
            ind = slice(_i, _i+1)

        elif isinstance(i, slice):
            ind = i
        else:
            raise IndexError("Index should be either an int or a slice.")

        return ind

    def _get_dict(self, i):

        ind = self._handle_index(i)

        data = {}
        # Fill coords to preserve any existing masks
        coords_filled = self.values.filled()
        for j, name in enumerate(self.names):
            data[name] = coords_filled[ind, j]

        if self.time is not None:
            data["time"] = self.time[ind]

        if self.origin is not None:
            data["origin"] = self.origin

        if self.frame is not None:
            data["frame"] = self.frame

        return data

    def __delitem__(self, i):

        ind = self._handle_index(i)

        for k, v in self.__dict__.items():
            if isinstance(v, np.ma.masked_array):
                self.__dict__[k] = np.delete(v, np.s_[ind], axis=0)
                self.__dict__[k].mask = np.delete(v.mask, np.s_[ind], axis=0)
            elif isinstance(v, np.ndarray):
                self.__dict__[k] = np.delete(v, np.s_[ind], axis=0)
            else:
                pass
        return

    @property
    def values(self):
        return self._values

    @property
    def covariance(self):
        return self._covariance

    @property
    def time(self):
        return self._time

    @property
    def origin(self):
        return self._origin

    @property
    def frame(self):
        return self._frame

    @property
    def names(self):
        return self._names

    def to_df(self,
            time_scale: str = "utc",
            include_frame: bool = False,
            include_origin: bool = False,
            include_masked: bool = False
        ) -> pd.DataFrame:
        """
        Represent coordinates as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : str
            If coordinates have an associated time, they will be stored in the
            dataframe as MJDs with this time scale.
        include_frame : bool
            Include the rotation frame (such as 'ecliptic' or 'equatorial').
        include_origin : bool
            Include the origin of the frame of reference.
        include_masked : bool
            Include columns that are fully masked.

        Returns
        -------
        df : `~pandas.DataFrame`
            DataFrame containing coordinates.
        """
        data = {}

        # Accessing semi-private astropy time function to set
        # the time scale to the one desired by the user
        if self.time is not None:
            time = deepcopy(self.time)
            time._set_scale(time_scale)
            data[f"mjd_{time.scale}"] = time.mjd

        for i, name in enumerate(self.names):
            coord = self.values[:, i]
            if not np.isnan(coord.filled()).all() or include_masked:
                data[name] = coord

        if self.covariance is not None:
            data["covariance"] = self.covariance

        if include_frame:
            data["frame"] = self.frame

        if include_origin:
            data["origin"] = self.origin

        df = pd.DataFrame(data)
        return df

class CartesianCoordinates(Coordinates):

    def __init__(
            self,
            x: Optional[np.ndarray] = None,
            y: Optional[np.ndarray] = None,
            z: Optional[np.ndarray] = None,
            vx: Optional[np.ndarray] = None,
            vy: Optional[np.ndarray] = None,
            vz: Optional[np.ndarray] = None,
            time: Optional[Time] = None,
            covariance: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic"
        ):
        """

        Parameters
        ----------
        x : `~numpy.ndarray` (N)
            X-coordinate in units of au.
        y : `~numpy.ndarray` (N)
            Y-coordinate in units of au.
        z : `~numpy.ndarray` (N)
            Z-coordinate in units of au.
        vx : `~numpy.ndarray` (N)
            X-coordinate velocity in in units of au per day.
        vy : `~numpy.ndarray` (N)
            Y-coordinate velocity in in units of au per day.
        vz : `~numpy.ndarray` (N)
            Z-coordinate velocity in in units of au per day.
        """
        coords = None
        for d, q in enumerate([x, y, z, vx, vy, vz]):
            coords = _ingest_coordinate(q, d, coords)

        self._x = coords[:, 0]
        self._y = coords[:, 1]
        self._z = coords[:, 2]
        self._vx = coords[:, 3]
        self._vy = coords[:, 4]
        self._vz = coords[:, 5]

        Coordinates.__init__(self, coords, covariance, time, origin, frame, CARTESIAN_COLS)
        return

    def __getitem__(self, i):
        data = self._get_dict(i)
        return CartesianCoordinates(**data)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def vx(self):
        return self._vx

    @property
    def vy(self):
        return self._vy

    @property
    def vz(self):
        return self._vz

class SphericalCoordinates(Coordinates):

    def __init__(
            self,
            rho: Optional[np.ndarray] = None,
            lon: Optional[np.ndarray] = None,
            lat: Optional[np.ndarray] = None,
            vrho: Optional[np.ndarray] = None,
            vlon: Optional[np.ndarray] = None,
            vlat: Optional[np.ndarray] = None,
            time: Optional[Time] = None,
            covariance: Optional[np.ndarray] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic"
        ):
        """

        Parameters
        ----------
        rho : `~numpy.ndarray` (N)
            Radial distance in units of au.
        lon : `~numpy.ndarray` (N)
            Longitudinal angle in units of degrees.
        lat : `~numpy.ndarray` (N)
            Latitudinal angle in units of degrees (geographic coordinate
            style with 0 degrees at the equator and ranging from -90 to 90).
        vrho : `~numpy.ndarray` (N)
            Radial velocity in units of au per day.
        vlon : `~numpy.ndarray` (N)
            Longitudinal velocity in units of degrees per day.
        vlat : `~numpy.ndarray` (N)
            Latitudinal velocity in units of degrees per day.
        """
        coords = None
        for d, q in enumerate([rho, lon, lat, vrho, vlon, vlat]):
            coords = _ingest_coordinate(q, d, coords)

        self._rho = coords[:, 0]
        self._lon = coords[:, 1]
        self._lat = coords[:, 2]
        self._vrho = coords[:, 3]
        self._vlon = coords[:, 4]
        self._vlat = coords[:, 5]

        Coordinates.__init__(self, coords, covariance, time, origin, frame, SPHERICAL_COLS)
        return

    def __getitem__(self, i):
        data = self._get_dict(i)
        return SphericalCoordinates(**data)

    @property
    def rho(self):
        return self._rho

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def vrho(self):
        return self._vrho

    @property
    def vlon(self):
        return self._vlon

    @property
    def vlat(self):
        return self._vlat