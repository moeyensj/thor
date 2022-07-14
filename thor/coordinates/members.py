import pandas as pd
from copy import deepcopy
from typing import Optional

from ..utils import Indexable
from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .spherical import SphericalCoordinates
from .keplerian import KeplerianCoordinates
from .cometary import CometaryCoordinates
from .transform import transform_coordinates

__all__ = ["CoordinateMembers"]

class CoordinateMembers(Indexable):

    def __init__(self,
            coordinates: Coordinates,
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
        else:
            err = (
                "coordinates should be one of:\n"
                "  CartesianCoordinates\n"
                "  SphericalCoordinates\n"
                "  KeplerianCoordinates\n"
                "  CometaryCoordinates\n"
            )
            raise TypeError(err)

        return

    def __len__(self):

        if self._cartesian is not None:
            N = len(self._cartesian)
        elif self._keplerian is not None:
            N = len(self._keplerian)
        elif self._cometary is not None:
            N = len(self._cometary)
        else: # self._spherical is not None:
            N = len(self._spherical)

        return N

    @property
    def cartesian(self):

        if self._cartesian is None:

            if self._keplerian is not None:
                self._cartesian = transform_coordinates(self._keplerian, "cartesian")
            elif self._cometary is not None:
                self._cartesian = transform_coordinates(self._cometary, "cartesian")
            elif self._spherical is not None:
                self._cartesian = transform_coordinates(self._spherical, "cartesian")

        return self._cartesian

    @property
    def spherical(self):

        if self._spherical is None:

            if self._cartesian is not None:
                self._spherical = transform_coordinates(self._cartesian, "spherical")
            elif self._keplerian is not None:
                self._spherical = transform_coordinates(self._keplerian, "spherical")
            elif self._cometary is not None:
                self._spherical = transform_coordinates(self._cometary, "spherical")

        return self._spherical

    @property
    def keplerian(self):

        if self._keplerian is None:

            if self._cartesian is not None:
                self._keplerian = transform_coordinates(self._cartesian, "keplerian")
            elif self._cometary is not None:
                self._keplerian = transform_coordinates(self._cometary, "keplerian")
            elif self._spherical is not None:
                self._keplerian = transform_coordinates(self._spherical, "keplerian")

        return self._keplerian

    @property
    def cometary(self):

        if self._cometary is None:

            if self._cartesian is not None:
                self._cometary = transform_coordinates(self._cartesian, "cometary")
            elif self._keplerian is not None:
                self._cometary = transform_coordinates(self._keplerian, "cometary")
            elif self._spherical is not None:
                self._cometary = transform_coordinates(self._spherical, "cometary")

        return self._cometary

    def to_df(self,
        time_scale: str = "tdb",
        coordinate_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Represent Orbits as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {"cartesian", "spherical", "keplerian", "cometary"}
            Desired output representation of the orbits.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing orbits.
        """
        if coordinate_type is None:
            coordinate_type_ = self.default_coordinate_type
        else:
            coordinate_type_ = coordinate_type

        if coordinate_type_ == "cartesian":
            df = self.cartesian.to_df(
                time_scale=time_scale
            )
        elif coordinate_type_ == "keplerian":
            df = self.keplerian.to_df(
                time_scale=time_scale
            )
        elif coordinate_type_ == "cometary":
            df = self.cometary.to_df(
                time_scale=time_scale
            )
        elif coordinate_type_ == "spherical":
            df = self.spherical.to_df(
                time_scale=time_scale
            )
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
