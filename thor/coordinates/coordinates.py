import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from copy import deepcopy
from typing import Optional, Union

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
CARTESIAN_UNITS = [u.au, u.au, u.au, u.au / u.d, u.au / u.d, u.au / u.d]

SPHERICAL_COLS = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
SPHERICAL_UNITS = [u.au, u.degree, u.degree, u.au / u.d, u.degree / u.d, u.degree / u.d]

__all__ = [
    "_ingest_coordinate",
    "CartesianCoordinates",
    "SphericalCoordinates"
]

def _ingest_coordinate(
    q: Union[list, np.ndarray], d: int, coords: Optional[np.ma.array] = None
):
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

class Coordinates:

    def to_df(self,
            time_scale: str = "utc",
            include_frame: bool = False,
            include_origin: bool = False
        ):
        """
        Represent coordinates as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : str
            If coordinates have an associated time, they will be stored in the
            dataframe as MJDs with this time scale.
        include_frame : str
            Include the rotation frame (such as 'ecliptic' or 'equatorial').
        include_origin : str
            Include the origin of the frame of reference.

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
            data[name] = self.coords[:, i]

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
            covariance: Optional[np.ndarray] = None,
            time: Optional[Time] = None,
            origin: str = "heliocentric",
            frame: str = "ecliptic"
        ):
        self.names = CARTESIAN_COLS

        coords = None
        for d, q in enumerate([x, y, z, vx, vy, vz]):
            coords = _ingest_coordinate(q, d, coords)

        self.coords = coords
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]
        self.vx = coords[:, 3]
        self.vy = coords[:, 4]
        self.vz = coords[:, 5]
        self.time = time
        self.covariance = covariance
        self.type = "cartesian"
        self.origin = origin
        self.frame = frame
        return

    def __getitem__(self, i):

        if isinstance(i, int):
            ind = slice(i, i+1)
        else:
            ind = i

        # Fill coords to preserve any existing masks
        coords_filled = self.coords.filled()
        data = {
            "x" : coords_filled[ind, 0],
            "y" : coords_filled[ind, 1],
            "z" : coords_filled[ind, 2],
            "vx" : coords_filled[ind, 3],
            "vy" : coords_filled[ind, 4],
            "vz" : coords_filled[ind, 5],
        }
        if self.time is not None:
            data["time"] = self.time[ind]

        if self.covariance is not None:
            data["covariance"] = self.covariance[ind]

        if self.origin is not None:
            data["origin"] = self.origin

        if self.frame is not None:
            data["frame"] = self.frame

        return CartesianCoordinates(**data)


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

        self.names = SPHERICAL_COLS

        coords = None
        for d, q in enumerate([rho, lon, lat, vrho, vlon, vlat]):
            coords = _ingest_coordinate(q, d, coords)

        self.coords = coords
        self.rho = coords[:, 0]
        self.lon = coords[:, 1]
        self.lat = coords[:, 2]
        self.vrho = coords[:, 3]
        self.vlon = coords[:, 4]
        self.vlat = coords[:, 5]
        self.time = time
        self.covariance = covariance
        self.type = "spherical"
        self.origin = origin
        self.frame = frame
        return

    def __getitem__(self, i):

        if isinstance(i, int):
            ind = slice(i, i+1)
        else:
            ind = i

        # Fill coords to preserve any existing masks
        coords_filled = self.coords.filled()
        data = {
            "rho" : coords_filled[ind, 0],
            "lon" : coords_filled[ind, 1],
            "lat" : coords_filled[ind, 2],
            "vrho" : coords_filled[ind, 3],
            "vlon" : coords_filled[ind, 4],
            "vlat" : coords_filled[ind, 5],
        }
        if self.time is not None:
            data["time"] = self.time[ind]

        if self.covariance is not None:
            data["covariance"] = self.covariance[ind]

        if self.origin is not None:
            data["origin"] = self.origin

        if self.frame is not None:
            data["frame"] = self.frame

        return SphericalCoordinates(**data)
