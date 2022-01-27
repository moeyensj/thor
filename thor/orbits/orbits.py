import uuid
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from astropy.time import Time
from ..utils import (
    Indexable,
    getHorizonsVectors
)
from ..coordinates import (
    CartesianCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
    transform_coordinates
)

logger = logging.getLogger(__name__)

__all__ = [
    "Orbits"
]


class Orbits(Indexable):

    def __init__(self,
            coordinates,
            ids=None,
            obj_ids=None,
        ):

        self._cartesian = None
        self._spherical = None
        self._keplerian = None

        if isinstance(coordinates, CartesianCoordinates):
            self._cartesian = deepcopy(coordinates)
        elif isinstance(coordinates, SphericalCoordinates):
            self._spherical = deepcopy(coordinates)
        elif isinstance(coordinates, KeplerianCoordinates):
            self._keplerian = deepcopy(coordinates)
        else:
            err = (
                "coordinates should be one of:\n"
                "  CartesianCoordinates\n"
                "  SphericalCoordinates\n"
                "  KeplerianCoordinates\n"
            )
            raise TypeError(err)

        if ids is not None:
            self._ids = ids
        else:
            self._ids = np.array([uuid.uuid4().hex for i in range(len(coordinates))])

        if obj_ids is not None:
            self._obj_ids = obj_ids
        else:
            self._obj_ids = np.array(["None" for i in range(len(coordinates))])

        return

    def __len__(self):

        if self._cartesian is not None:
            N = len(self._cartesian)
        elif self._keplerian is not None:
            N = len(self._keplerian)
        else: # self._spherical is not None:
            N = len(self._spherical)

        return N

    @property
    def ids(self):
        return self._ids

    @property
    def obj_ids(self):
        return self._obj_ids

    @property
    def cartesian(self):

        if self._cartesian is None:

            if self._keplerian is not None:
                self._cartesian = transform_coordinates(self._keplerian, "cartesian")
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

        return self._spherical

    @property
    def keplerian(self):

        if self._keplerian is None:

            if self._cartesian is not None:
                self._keplerian = transform_coordinates(self._cartesian, "keplerian")
            elif self._spherical is not None:
                self._keplerian = transform_coordinates(self._spherical, "keplerian")

        return self._keplerian

    @classmethod
    def from_horizons(cls, ids, times):

        assert len(times) == 1

        vectors = getHorizonsVectors(
            ids,
            times,
            location="@sun",
            id_type="smallbody",
            aberrations="geometric",
        )

        coordinates = CartesianCoordinates(
            times=Time(
                vectors["datetime_jd"].values,
                scale="tdb",
                format="jd"
            ),
            x=vectors["x"].values,
            y=vectors["y"].values,
            z=vectors["z"].values,
            vx=vectors["vx"].values,
            vy=vectors["vy"].values,
            vz=vectors["vz"].values,
            origin="heliocenter",
            frame="ecliptic"
        )
        obj_ids = vectors["targetname"].values

        return cls(coordinates, obj_ids=obj_ids)

    def to_df(self,
            time_scale: str = "tdb",
            coordinate_type: str = "cartesian",
        ) -> pd.DataFrame:
        """
        Represent Orbits as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {"cartesian", "spherical", "keplerian"}
            Desired output representation of the orbits.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing orbits.
        """
        if coordinate_type == "cartesian":
            df = self.cartesian.to_df(
                time_scale=time_scale
            )
        elif coordinate_type == "keplerian":
            df = self.keplerian.to_df(
                time_scale=time_scale
            )
        elif coordinate_type == "spherical":
            df = self.spherical.to_df(
                time_scale=time_scale
            )
        else:
            err = (
                "coordinate_type should be one of:\n"
                "  cartesian\n"
                "  spherical\n"
                "  keplerian\n"
            )
            raise ValueError(err)

        df.insert(0, "orbit_id", self.ids)
        df.insert(1, "obj_id", self.obj_ids)

        return df
