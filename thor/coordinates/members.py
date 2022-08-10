import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional
from collections import OrderedDict

from ..utils import Indexable
from .coordinates import Coordinates
from .cartesian import (
    CartesianCoordinates,
    CARTESIAN_COLS,
)
from .spherical import (
    SphericalCoordinates,
    SPHERICAL_COLS,
)
from .keplerian import (
    KeplerianCoordinates,
    KEPLERIAN_COLS,
)
from .cometary import (
    CometaryCoordinates,
    COMETARY_COLS,
)
from .transform import transform_coordinates

__all__ = ["CoordinateMembers"]

class CoordinateMembers(Indexable):

    def __init__(self,
            coordinates: Optional[Coordinates] = None,
            cartesian=True,
            keplerian=True,
            spherical=True,
            cometary=True

    ):
        self._cartesian = None
        self._spherical = None
        self._keplerian = None
        self._cometary = None
        self.default_coordinate_type = None

        allowed_coordinate_types = set()
        if cartesian:
            allowed_coordinate_types.add("CartesianCoordinates")
        if spherical:
            allowed_coordinate_types.add("SphericalCoordinates")
        if keplerian:
            allowed_coordinate_types.add("KeplerianCoordinates")
        if cometary:
            allowed_coordinate_types.add("CometaryCoordinates")
        self.__allowed_coordinate_types = allowed_coordinate_types

        if isinstance(coordinates, CartesianCoordinates) and cartesian:
            self._cartesian = deepcopy(coordinates)
            self.default_coordinate_type = "cartesian"
        elif isinstance(coordinates, SphericalCoordinates) and spherical:
            self._spherical = deepcopy(coordinates)
            self.default_coordinate_type = "spherical"
        elif isinstance(coordinates, KeplerianCoordinates) and keplerian:
            self._keplerian = deepcopy(coordinates)
            self.default_coordinate_type = "keplerian"
        elif isinstance(coordinates, CometaryCoordinates) and cometary:
            self._cometary = deepcopy(coordinates)
            self.default_coordinate_type = "cometary"
        elif coordinates is None:
            pass
        else:
            err = "coordinates should be one of:\n"
            err += "".join([f"  {type_i}\n" for type_i in list(self.__allowed_coordinate_types)])
            raise TypeError(err)

        if coordinates is not None:
            index = np.arange(0, len(coordinates), 1)
        else:
            index = np.array([])

        Indexable.__init__(self, index)
        return

    @property
    def cartesian(self):

        if "CartesianCoordinates" not in self.__allowed_coordinate_types:
            err = ("Cartesian coordinates are not supported by this class.")
            raise ValueError(err)

        if self._cartesian is None and "CartesianCoordinates":

            if self._keplerian is not None and "KeplerianCoordinates" in self.__allowed_coordinate_types:
                self._cartesian = transform_coordinates(self._keplerian, "cartesian")
            elif self._cometary is not None and "CometaryCoordinates" in self.__allowed_coordinate_types:
                self._cartesian = transform_coordinates(self._cometary, "cartesian")
            elif self._spherical is not None and "SphericalCoordinates" in self.__allowed_coordinate_types:
                self._cartesian = transform_coordinates(self._spherical, "cartesian")

        return self._cartesian

    @property
    def spherical(self):

        if "SphericalCoordinates" not in self.__allowed_coordinate_types:
            err = ("Spherical coordinates are not supported by this class.")
            raise ValueError(err)

        if self._spherical is None:

            if self._cartesian is not None and "CartesianCoordinates" in self.__allowed_coordinate_types:
                self._spherical = transform_coordinates(self._cartesian, "spherical")
            elif self._keplerian is not None and "KeplerianCoordinates" in self.__allowed_coordinate_types:
                self._spherical = transform_coordinates(self._keplerian, "spherical")
            elif self._cometary is not None and "CometaryCoordinates" in self.__allowed_coordinate_types:
                self._spherical = transform_coordinates(self._cometary, "spherical")

        return self._spherical

    @property
    def keplerian(self):

        if "KeplerianCoordinates" not in self.__allowed_coordinate_types:
            err = ("Keplerian coordinates are not supported by this class.")
            raise ValueError(err)

        if self._keplerian is None:

            if self._cartesian is not None and "CartesianCoordinates" in self.__allowed_coordinate_types:
                self._keplerian = transform_coordinates(self._cartesian, "keplerian")
            elif self._cometary is not None and "CometaryCoordinates" in self.__allowed_coordinate_types:
                self._keplerian = transform_coordinates(self._cometary, "keplerian")
            elif self._spherical is not None and "SphericalCoordinates" in self.__allowed_coordinate_types:
                self._keplerian = transform_coordinates(self._spherical, "keplerian")

        return self._keplerian

    @property
    def cometary(self):

        if "CometaryCoordinates" not in self.__allowed_coordinate_types:
            err = ("Keplerian coordinates are not supported by this class.")
            raise ValueError(err)

        if self._cometary is None:

            if self._cartesian is not None and "CartesianCoordinates" in self.__allowed_coordinate_types:
                self._cometary = transform_coordinates(self._cartesian, "cometary")
            elif self._keplerian is not None and "KeplerianCoordinates" in self.__allowed_coordinate_types:
                self._cometary = transform_coordinates(self._keplerian, "cometary")
            elif self._spherical is not None and "SphericalCoordinates" in self.__allowed_coordinate_types:
                self._cometary = transform_coordinates(self._spherical, "cometary")

        return self._cometary

    def to_df(self,
            time_scale: str = "tdb",
            coordinate_type: Optional[str] = None,
            sigmas: bool = False,
            covariances: bool = False,
        ) -> pd.DataFrame:
        """
        Represent coordinates as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {None, "cartesian", "spherical", "keplerian", "cometary"}
            Desired output representation of the coordinates. If None, will default
            to coordinate type that was given at class initialization.
        sigmas : bool, optional
            Include 1-sigma uncertainty columns.
        covariances : bool, optional
            Include lower triangular covariance matrix columns.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing coordinates.

        Raises
        ------
        ValueError: If coordinate_type is not one of {'cartesian', 'keplerian',
            'cometary', 'spherical'}.
        """
        if coordinate_type is None:
            coordinate_type_ = self.default_coordinate_type
        else:
            coordinate_type_ = coordinate_type

        kwargs = {
            "time_scale" : time_scale,
            "sigmas" : sigmas,
            "covariances" : covariances
        }
        if coordinate_type_ == "cartesian":
            df = self.cartesian.to_df(**kwargs)
        elif coordinate_type_ == "keplerian":
            df = self.keplerian.to_df(**kwargs)
        elif coordinate_type_ == "cometary":
            df = self.cometary.to_df(**kwargs)
        elif coordinate_type_ == "spherical":
            df = self.spherical.to_df(**kwargs)
        else:
            err = (
                "coordinate_type should be one of:\n"
                "  cartesian\n"
                "  spherical\n"
                "  keplerian\n"
                "  cometary\n"
            )
            raise ValueError(err)

        return df

    @staticmethod
    def _dict_from_df(
            df: pd.DataFrame,
            cartesian: bool = True,
            keplerian: bool = True,
            cometary: bool = True,
            spherical: bool = True,
            coord_cols: Optional[OrderedDict] = None,
            origin_col: str = "origin",
            frame_col: str = "frame"
        ) -> dict:
        """
        Create a dictionary with a single instance of coordinates
        from a `pandas.DataFrame`. If all coordinate types are set to true,
        this function will look for coordinates in this order: Cartesian,
        Keplerian, Cometary, Spherical.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing coordinates of a single type.
        cartesian : bool, optional
            Look for Cartesian coordinates.
        keplerian : bool, optional
            Look for Keplerian coordinates.
        cometary : bool, optional
            Look for Cometary coordinates.
        spherical : bool, optional
            Look for Spherical coordinates.
        coord_cols : OrderedDict, optional
            Ordered dictionary containing the coordinate dimensions as keys and their equivalent columns
            as values. If None, this function will use the default dictionaries for each coordinate class.
            The following coordinate (dictionary) keys are supported:
                Cartesian columns: x, y, z, vx, vy, vz
                Keplerian columns: a, e, i, raan, ap, M
                Cometary columns: q, e, i, raan, ap, tp
                Spherical columns: rho, lon, lat, vrho, vlon, vlat
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.

        Returns
        -------
        data : dict
            Dictionary containing coordinates extracted from the given `~pandas.DataFrame`.
        """
        data = {}
        columns = df.columns.values
        if coord_cols is None:
            if cartesian and np.all(np.in1d(list(CARTESIAN_COLS.values()), columns)):
                coord_class = CartesianCoordinates
                coord_cols_ = CARTESIAN_COLS
            elif keplerian and np.all(np.in1d(list(KEPLERIAN_COLS.values()), columns)):
                coord_class = KeplerianCoordinates
                coord_cols_ = KEPLERIAN_COLS
            elif cometary and np.all(np.in1d(list(COMETARY_COLS.values()), columns)):
                coord_class = CometaryCoordinates
                coord_cols_ = COMETARY_COLS
            elif spherical and np.all(np.in1d(list(SPHERICAL_COLS.values()), columns)):
                coord_class = SphericalCoordinates
                coord_cols_ = SPHERICAL_COLS
            else:
                err = "No supported coordinates could be found in the given dataframe.\n" \
                    "The following coordinate columns were searched for:\n"
                if cartesian:
                    err += f" Cartesian columns: {', '.join(CARTESIAN_COLS.keys())}\n"
                if keplerian:
                    err += f" Keplerian columns: {', '.join(KEPLERIAN_COLS.keys())}\n"
                if cometary:
                    err += f" Cometary columns: {', '.join(COMETARY_COLS.keys())}\n"
                if spherical:
                    err += f" Spherical columns: {', '.join(SPHERICAL_COLS.keys())}\n"
                raise ValueError(err)

        elif isinstance(coord_cols, OrderedDict):
            if cartesian and coord_cols.keys() == CARTESIAN_COLS.keys():
                coord_class = CartesianCoordinates
            elif keplerian and coord_cols.keys() == KEPLERIAN_COLS.keys():
                coord_class = KeplerianCoordinates
            elif cometary and coord_cols.keys() == COMETARY_COLS.keys():
                coord_class = CometaryCoordinates
            elif spherical and coord_cols.keys() == SPHERICAL_COLS.keys():
                coord_class = SphericalCoordinates
            else:
                err = "Coordinate column dictionary keys could not be\n" \
                    "matched to any of the supported coordinates for this class:\n"
                if cartesian:
                    err += f" Cartesian columns: {', '.join(CARTESIAN_COLS.keys())}\n"
                if keplerian:
                    err += f" Keplerian columns: {', '.join(KEPLERIAN_COLS.keys())}\n"
                if cometary:
                    err += f" Cometary columns: {', '.join(COMETARY_COLS.keys())}\n"
                if spherical:
                    err += f" Spherical columns: {', '.join(SPHERICAL_COLS.keys())}\n"
                raise ValueError(err)

            coord_cols_ = coord_cols

        else:
            err = ("coord_cols should be one of {None, OrderedDict}.")
            raise ValueError(err)

        coordinates = coord_class.from_df(
            df,
            coord_cols=coord_cols_,
            origin_col=origin_col,
            frame_col=frame_col
        )
        data["coordinates"] = coordinates

        return data

    @classmethod
    def from_df(
            cls: "CoordinateMembers",
            df: pd.DataFrame,
            cartesian: bool = True,
            keplerian: bool = True,
            cometary: bool = True,
            spherical: bool = True,
            coord_cols: Optional[OrderedDict] = None,
            origin_col: str = "origin",
            frame_col: str = "frame"
        ) -> "CoordinateMembers":
        """
        Instantiate CoordinateMembers from a `pandas.DataFrame`. If all
        coordinate types are set to true,  this function will look
        for coordinates in this order: Cartesian, Keplerian, Cometary, Spherical.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing coordinates of a single type.
        cartesian : bool, optional
            Look for Cartesian coordinates.
        keplerian : bool, optional
            Look for Keplerian coordinates.
        cometary : bool, optional
            Look for Cometary coordinates.
        spherical : bool, optional
            Look for Spherical coordinates.
        coord_cols : OrderedDict, optional
            Ordered dictionary containing the coordinate dimensions as keys and their equivalent columns
            as values. If None, this function will use the default dictionaries for each coordinate class.
            The following coordinate (dictionary) keys are supported:
                Cartesian columns: x, y, z, vx, vy, vz
                Keplerian columns: a, e, i, raan, ap, M
                Cometary columns: q, e, i, raan, ap, tp
                Spherical columns: rho, lon, lat, vrho, vlon, vlat
        origin_col : str
            Name of the column containing the origin of each coordinate.
        frame_col : str
            Name of the column containing the coordinate frame.

        Returns
        -------
        cls : `~thor.coordinates.members.Members`
            CoordinateMembers extracted from the given `~pandas.DataFrame`.
        """
        data = cls._dict_from_df(
            df,
            cartesian=cartesian,
            keplerian=keplerian,
            cometary=cometary,
            spherical=spherical,
            coord_cols=coord_cols,
            origin_col=origin_col,
            frame_col=frame_col
        )
        return cls(**data)