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
from ..coordinates.covariances import (
    sigmas_to_covariance,
    sample_covariance
)
from ..coordinates.members import CoordinateMembers
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

    def generate_variants(
            self,
            num_samples: int = 1000,
            percent: float = 0.1
        ) -> "Orbits":
        """
        Generate variants of the orbits sampled from their covariance matrices. If the covariances
        matrices are fully masked, then a percentage of the state is used to populate the diagonals
        of each covariance matrix.

        Parameters
        ----------
        num_samples : int
            The number of variants to generate.
        percent : float
            The percentage of the state to use to populate the diagonals of the covariance matrices.

        Returns
        -------
        variants : Orbits
            The generated variants.
        """
        kwargs = {}
        if self.default_coordinate_type == "cartesian":
            means = self.cartesian.values
            covariances = self.cartesian.covariances
            times = self.cartesian.times.tdb.mjd
            origin = self.cartesian.origin

            kwargs["frame"] = self.cartesian.frame
            kwargs["names"] = self.cartesian.names
            kwargs["units"] = self.cartesian.units

        elif self.default_coordinate_type == "spherical":
            means = self.spherical.values
            covariances = self.spherical.covariances
            times = self.spherical.times.tdb.mjd
            origin = self.spherical.origin

            kwargs["frame"] = self.spherical.frame
            kwargs["names"] = self.spherical.names
            kwargs["units"] = self.spherical.units

        elif self.default_coordinate_type == "keplerian":
            means = self.keplerian.values
            covariances = self.keplerian.covariances
            times = self.keplerian.times.tdb.mjd
            origin = self.keplerian.origin

            kwargs["frame"] = self.keplerian.frame
            kwargs["names"] = self.keplerian.names
            kwargs["units"] = self.keplerian.units

        elif self.default_coordinate_type == "cometary":
            means = self.cometary.values
            covariances = self.cometary.covariances
            times = self.cometary.times.tdb.mjd
            origin = self.cometary.origin

            kwargs["frame"] = self.cometary.frame
            kwargs["names"] = self.cometary.names
            kwargs["units"] = self.cometary.units

        variants = np.zeros((num_samples * len(means), 6), dtype=np.float64)
        orbit_ids = []
        object_ids = []
        times_mjd_tdb = []
        origins = []
        for i, (orbit_id, object_id, mean, covariance, time, origin_i) in enumerate(
                zip(self.orbit_ids, self.object_ids, means, covariances, times, origin)
            ):

            if np.all(covariance.mask):
                logger.info(f"{orbit_id} has a fully masked covariance matrix. Using {percent * 100}% of the state to populate the diagonals.")
                covariance_i = sigmas_to_covariance(
                    percent * means[i:i+1]
                )[0]
            else:
                covariance_i = covariance

            variants[i * num_samples:(i + 1) * num_samples] = sample_covariance(
                mean,
                covariance_i,
                num_samples=num_samples
            )
            orbit_ids.append(np.hstack([orbit_id for i in range(num_samples)]))
            object_ids.append(np.hstack([object_id for i in range(num_samples)]))
            times_mjd_tdb.append(np.hstack([time for i in range(num_samples)]))
            origins.append(np.hstack([origin_i for i in range(num_samples)]))

        orbit_ids = np.hstack(orbit_ids)
        object_ids = np.hstack(object_ids)
        times = Time(
            np.hstack(times_mjd_tdb),
            scale="tdb",
            format="mjd"
        )
        origins = np.hstack(origins)

        if self.default_coordinate_type == "cartesian":
            coordinates = CartesianCoordinates(
                x=variants[:, 0],
                y=variants[:, 1],
                z=variants[:, 2],
                vx=variants[:, 3],
                vy=variants[:, 4],
                vz=variants[:, 5],
                times=times,
                covariances=None,
                origin=origins,
                **kwargs
            )

        elif self.default_coordinate_type == "spherical":
            coordinates = SphericalCoordinates(
                rho=variants[:, 0],
                lon=variants[:, 1],
                lat=variants[:, 2],
                vrho=variants[:, 3],
                vlon=variants[:, 4],
                vlat=variants[:, 5],
                times=times,
                covariances=None,
                origin=origins,
                **kwargs
            )

        elif self.default_coordinate_type == "keplerian":
            coordinates = KeplerianCoordinates(
                a=variants[:, 0],
                e=variants[:, 1],
                i=variants[:, 2],
                raan=variants[:, 3],
                ap=variants[:, 4],
                M=variants[:, 5],
                times=times,
                covariances=None,
                origin=origins,
                **kwargs
            )

        elif self.default_coordinate_type == "cometary":
            coordinates = CometaryCoordinates(
                q=variants[:, 0],
                e=variants[:, 1],
                i=variants[:, 2],
                raan=variants[:, 3],
                ap=variants[:, 4],
                tp=variants[:, 5],
                times=times,
                covariances=None,
                origin=origins,
                **kwargs
            )

        return Orbits(coordinates, orbit_ids=orbit_ids, object_ids=object_ids)

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
