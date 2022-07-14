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
from ..coordinates.members import CoordinateMembers
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from .classification import calc_orbit_class

logger = logging.getLogger(__name__)

__all__ = [
    "Orbits",
    "FittedOrbits"
]


class Orbits(CoordinateMembers):

    def __init__(self,
            coordinates,
            ids=None,
            object_ids=None,
            classes=None,
        ):
        CoordinateMembers.__init__(self,
            coordinates=coordinates,
            cartesian=True,
            keplerian=True,
            spherical=True,
            cometary=True
        )
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

        Indexable.__init__(self, index=self.ids)
        return

    @property
    def ids(self):
        return self._ids

    @property
    def object_ids(self):
        return self._object_ids

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

        elif coordinate_type == "cometary":
            elements = get_Horizons_elements(
                ids,
                times,
                location="@sun",
                id_type="smallbody",
            )

            tp = Time(
                elements["Tp_jd"].values,
                scale="tdb",
                format="jd"
            )
            coordinates = CometaryCoordinates(
                times=Time(
                    elements["datetime_jd"].values,
                    scale="tdb",
                    format="jd"
                ),
                q=elements["q"].values,
                e=elements["e"].values,
                i=elements["incl"].values,
                raan=elements["Omega"].values,
                ap=elements["w"].values,
                tp=tp.tdb.mjd,
                origin="heliocenter",
                frame="ecliptic"
            )
            object_ids = elements["targetname"].values

        else:
            err = (
                "coordinate_type should be one of {'cartesian', 'keplerian', 'cometary'}"
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
        df = CoordinateMembers.to_df(self,
            time_scale=time_scale,
            coordinate_type=coordinate_type
        )
        df.insert(0, "orbit_id", self.ids)
        df.insert(1, "object_id", self.object_ids)
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
