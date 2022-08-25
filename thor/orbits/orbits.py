import uuid
import logging
import numpy as np
import pandas as pd
from astropy.time import Time
from typing import (
    List,
    Optional,
)
from collections import OrderedDict

from ..utils.indexable import Indexable
from ..utils.horizons import (
    get_Horizons_vectors,
    get_Horizons_elements
)
from ..utils.sbdb import (
    convert_SBDB_covariances,
    get_SBDB_elements
)
from ..coordinates.members import CoordinateMembers
from ..coordinates.covariances import sigmas_to_covariance
from ..coordinates.cartesian import CartesianCoordinates
from ..coordinates.keplerian import KeplerianCoordinates
from ..coordinates.cometary import CometaryCoordinates
from ..coordinates.spherical import SphericalCoordinates
from .classification import calc_orbit_class

logger = logging.getLogger(__name__)

__all__ = [
    "Orbits",
    "FittedOrbits"
]


class Orbits(CoordinateMembers):

    def __init__(self,
            coordinates,
            orbit_ids=None,
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
        if orbit_ids is not None:
            self._orbit_ids = orbit_ids
        else:
            self._orbit_ids = np.array([uuid.uuid4().hex for i in range(len(coordinates))])

        if object_ids is not None:
            self._object_ids = object_ids
        else:
            self._object_ids = np.array(["None" for i in range(len(coordinates))])

        if classes is not None:
            self._classes = classes
        else:
            self._classes = None

        Indexable.__init__(self, index="orbit_ids")
        return

    @property
    def orbit_ids(self):
        return self._orbit_ids

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def classes(self):

        if self._classes is None:
            self._classes = calc_orbit_class(self.keplerian)

        return self._classes

    @classmethod
    def from_SBDB(cls,
            ids: List,
        ):
        """
        Query JPL's Small-Body Database (SBDB) for orbits. The epoch at
        which the orbits are returned are near the epoch as published by the
        Minor Planet Center.

        By default, the orbit's covariance matrices are also queried for. If they
        are not available, then the 1-sigma uncertainties are used to construct
        the covariance matrices.

        Parameters
        ----------
        ids : list
            List of object IDs to query.

        Returns
        -------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits object containing the queried orbits.
        """
        results = get_SBDB_elements(ids)

        object_ids = []
        classes = []
        coords_cometary = np.zeros((len(results), 6), dtype=np.float64)
        covariances_sbdb = np.zeros((len(results), 6, 6), dtype=np.float64)
        times = np.zeros((len(results)), dtype=np.float64)

        for i, result in enumerate(results):
            object_ids.append(result["object"]["fullname"])
            classes.append(result["object"]["orbit_class"]["code"])

            if "covariance" in result["orbit"]:

                result_i  = result["orbit"]["covariance"]

                coords_cometary[i, 0] = result_i["elements"]["q"].value
                coords_cometary[i, 1] = result_i["elements"]["e"]
                coords_cometary[i, 2] = result_i["elements"]["i"].value
                coords_cometary[i, 3] = result_i["elements"]["om"].value
                coords_cometary[i, 4] = result_i["elements"]["w"].value
                coords_cometary[i, 5] = Time(result_i["elements"]["tp"].value, scale="tdb", format="jd").mjd

                covariances_sbdb[i, :, :] = result_i["data"]

                times[i] = result_i["epoch"].value

            else:

                result_i  = result["orbit"]

                coords_cometary[i, 0] = result_i["elements"]["q"].value
                coords_cometary[i, 1] = result_i["elements"]["e"]
                coords_cometary[i, 2] = result_i["elements"]["i"].value
                coords_cometary[i, 3] = result_i["elements"]["om"].value
                coords_cometary[i, 4] = result_i["elements"]["w"].value
                coords_cometary[i, 5] = Time(result_i["elements"]["tp"].value, scale="tdb", format="jd").mjd

                sigmas = np.array([[
                    result_i["elements"]["e_sig"], result_i["elements"]["q_sig"].value, result_i["elements"]["tp_sig"].value,
                    result_i["elements"]["om_sig"].value, result_i["elements"]["w_sig"].value, result_i["elements"]["i_sig"].value
                ]])

                covariances_sbdb[i, :, :] = sigmas_to_covariance(sigmas).filled()[0]

                times[i] = result_i["epoch"].value

        covariances_cometary = convert_SBDB_covariances(covariances_sbdb)
        times = Time(times, scale="tdb", format="jd")

        coordinates = CometaryCoordinates(
            times=times,
            q=coords_cometary[:, 0],
            e=coords_cometary[:, 1],
            i=coords_cometary[:, 2],
            raan=coords_cometary[:, 3],
            ap=coords_cometary[:, 4],
            tp=coords_cometary[:, 5],
            covariances=covariances_cometary)

        object_ids = np.array(object_ids)
        classes = np.array(classes)

        return cls(coordinates, object_ids=object_ids, classes=classes)

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

    def to_frame(self, frame: str):
        """
        Rotate coordinates to desired frame. Rotation is applied to the Cartesian coordinates.
        All other coordinate types are reset to None.

        Parameters
        ----------
        frame : {'ecliptic', 'equatorial'}
            Desired reference frame of the output coordinates.
        """
        self._cartesian = self.cartesian.to_frame(frame)
        self._keplerian = None
        self._cometary = None
        self._spherical = None
        return

    def to_origin(self, origin: str):
        """
        Translate coordinates to a different origin. Translation is applied to the Cartesian coordinates.
        All other coordinate types are reset to None.

        Parameters
        ----------
        origin : {'heliocenter', 'barycenter'}
            Name of the desired origin.
        """
        self._cartesian = self.cartesian.to_origin(origin)
        self._keplerian = None
        self._cometary = None
        self._spherical = None
        return

    def to_df(
            self,
            time_scale: str = "tdb",
            coordinate_type: Optional[str] = None,
            sigmas: bool = False,
            covariances: bool = False,
        ) -> pd.DataFrame:
        """
        Represent Orbits as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {"cartesian", "spherical", "keplerian", "cometary"}
            Desired output representation of the orbits.
        sigmas : bool, optional
            Include 1-sigma uncertainty columns.
        covariances : bool, optional
            Include lower triangular covariance matrix columns.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing orbits.
        """
        df = CoordinateMembers.to_df(
            self,
            time_scale=time_scale,
            coordinate_type=coordinate_type,
            sigmas=sigmas,
            covariances=covariances
        )
        df.insert(0, "orbit_id", self.orbit_ids)
        df.insert(1, "object_id", self.object_ids)
        if self._classes is not None:
            df.insert(len(df.columns), "class", self.classes)

        return df

    @classmethod
    def from_df(
            cls: "Orbits",
            df: pd.DataFrame,
            coord_cols: Optional[OrderedDict] = None,
            origin_col: str = "origin",
            frame_col: str = "frame"
        ) -> "Orbits":
        """
        Read Orbits class from a `~pandas.DataFrame`.

        Parameters
        ----------
        df : `~pandas.DataFrame`
            DataFrame containing orbits.
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
        cls : `~thor.orbits.orbits.Orbits`
            Orbits class.
        """
        data = cls._dict_from_df(
            df,
            cartesian=True,
            keplerian=True,
            cometary=True,
            spherical=True,
            coord_cols=coord_cols,
            origin_col=origin_col,
            frame_col=frame_col
        )

        columns = df.columns.values
        if "orbit_id" in columns:
            data["orbit_ids"] = df["orbit_id"].values
        else:
            data["orbit_ids"] = None

        if "object_id" in columns:
            data["object_ids"] = df["object_id"].values
        else:
            data["object_ids"] = None

        if "class" in columns:
            data["classes"] = df["class"].values
        else:
            data["classes"] = None

        return cls(**data)

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
