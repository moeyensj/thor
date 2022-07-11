import uuid
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from astropy.time import Time
from typing import (
    List,
    Optional
)

from ..utils.indexable import Indexable
from ..utils.horizons import (
    get_Horizons_vectors,
    get_Horizons_elements
)
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.spherical import SphericalCoordinates
from ..coordinates.transform import transform_coordinates
from .classification import calc_orbit_class

logger = logging.getLogger(__name__)

__all__ = [
    "Orbits",
    "FittedOrbits"
]


class Orbits(Indexable):

    def __init__(self,
            coordinates,
            ids=None,
            object_ids=None,
            classes=None,
        ):

        self._cartesian = None
        self._spherical = None
        self._keplerian = None
        self._cometary = None
        self.default_coordinate_type = None

        if isinstance(coordinates, CartesianCoordinates):
            self._cartesian = deepcopy(coordinates)
            self.default_coordinate_type = "cartesian"
        elif isinstance(coordinates, SphericalCoordinates):
            self._spherical = deepcopy(coordinates)
            self.default_coordinate_type = "spherical"
        elif isinstance(coordinates, KeplerianCoordinates):
            self._keplerian = deepcopy(coordinates)
            self.default_coordinate_type = "keplerian"
        elif isinstance(coordinates, CometaryCoordinates):
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

        if ids is not None:
            self._ids = ids
        else:
            self._ids = np.array([uuid.uuid4().hex for i in range(len(coordinates))])

        if object_ids is not None:
            self._object_ids = object_ids
        else:
            self._object_ids = np.array(["None" for i in range(len(coordinates))])

        if classes is not None:
            self._classes = classes
        else:
            self._classes = None

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
    def ids(self):
        return self._ids

    @property
    def object_ids(self):
        return self._object_ids

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

    @property
    def classes(self):

        if self._classes is None:
            self._classes = calc_orbit_class(self.keplerian)

        return self._classes

    @classmethod
    def from_Horizons(cls,
            ids: List,
            times: Time,
            coordinate_type="cartesian"
        ):
        assert len(times) == 1

        if coordinate_type == "cartesian":
            vectors = get_Horizons_vectors(
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
            object_ids = vectors["targetname"].values

        elif coordinate_type == "keplerian":
            elements = get_Horizons_elements(
                ids,
                times,
                location="@sun",
                id_type="smallbody",
            )

            coordinates = KeplerianCoordinates(
                times=Time(
                    elements["datetime_jd"].values,
                    scale="tdb",
                    format="jd"
                ),
                a=elements["a"].values,
                e=elements["e"].values,
                i=elements["incl"].values,
                raan=elements["Omega"].values,
                ap=elements["w"].values,
                M=elements["M"].values,
                origin="heliocenter",
                frame="ecliptic"
            )
            object_ids = elements["targetname"].values

        else:
            err = (
                "coordinate_type should be one of {'cartesian', 'keplerian'}"
            )
            raise ValueError(err)

        return cls(coordinates, object_ids=object_ids)

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

        df.insert(0, "orbit_id", self.ids)
        df.insert(1, "obj_id", self.object_ids)
        if self._classes is not None:
            df.insert(len(df.columns), "class", self.classes)

        return df

class FittedOrbits(Orbits):

    def __init__(self,
            coordinates,
            ids=None,
            object_ids=None,
            members=None,
            num_obs=None,
            arc_length=None,
            chi2=None,
            rchi2=None,
        ):
        Orbits.__init__(
            self,
            coordinates=coordinates,
            ids=ids,
            object_ids=object_ids,
        )

        N = len(self)
        if members is None:
            self._members = [np.array([]) for i in range(N)]
        else:
            assert len(members) == N
            self._members = members

        if num_obs is None:
            self._num_obs = np.zeros(N, dtype=int)
        else:
            assert len(num_obs) == N
            self._num_obs = num_obs

        if arc_length is None:
            self._arc_length = np.zeros(N, dtype=float)
        else:
            assert len(arc_length) == N
            self._arc_length = arc_length

        if chi2 is None:
            self._chi2 = np.array([np.NaN for i in range(N)])
        else:
            assert len(chi2) == N
            self._chi2 = chi2

        if rchi2 is None:
            self._rchi2 = np.array([np.NaN for i in range(N)])
        else:
            assert len(rchi2) == N
            self._rchi2 = rchi2

        return

    @property
    def members(self):
        return self._members

    @property
    def num_obs(self):
        return self._num_obs

    @property
    def arc_length(self):
        return self._arc_length

    @property
    def chi2(self):
        return self._chi2

    @property
    def rchi2(self):
        return self._rchi2

    def to_df(self, time_scale="tdb", coordinate_type="cartesian"):

        df = Orbits.to_df(self,
            time_scale=time_scale,
            coordinate_type=coordinate_type
        )
        df.insert(len(df.columns), "num_obs", self.num_obs)
        df.insert(len(df.columns), "arc_length", self.arc_length)
        df.insert(len(df.columns), "chi2", self.chi2)
        df.insert(len(df.columns), "rchi2", self.rchi2)
        return df
