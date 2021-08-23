import ast
import logging
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from ..utils import _checkTime
from ..utils import getHorizonsVectors
from .kepler import convertOrbitalElements

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
CARTESIAN_UNITS = [u.au, u.au, u.au, u.au / u.d,  u.au / u.d,  u.au / u.d]
KEPLERIAN_COLS = ["a", "e", "i", "omega", "w", "M"]
KEPLERIAN_UNITS = [u.au, u.dimensionless_unscaled, u.deg, u.deg, u.deg, u.deg]

logger = logging.getLogger(__name__)

__all__ = [
    "Orbits"
]

def _convertOrbitUnits(
        orbits,
        orbit_units,
        desired_units
    ):
    orbits_ = orbits.copy()
    for i, (unit_desired, unit_given) in enumerate(zip(desired_units, orbit_units)):
        if unit_desired != unit_given:
            orbits_[:, i] = orbits_[:, i] * unit_given.to(unit_desired)

    return orbits_

def _convertCovarianceUnits(
        covariances,
        orbit_units,
        desired_units,
    ):
    orbit_units_ = np.array([orbit_units])
    covariance_units = np.dot(
        orbit_units_.T,
        orbit_units_
    )

    desired_units_ = np.array([desired_units])
    desired_units_ = np.dot(
        desired_units_.T,
        desired_units_
    )

    covariances_ = np.stack(covariances)
    for i in range(6):
        for j in range(6):
            unit_desired = desired_units_[i, j]
            unit_given = covariance_units[i, j]
            if unit_desired != unit_given:
                covariances_[:, i, j] = covariances_[:, i, j] * unit_given.to(unit_desired)

    covariances_ = np.split(covariances_, covariances_.shape[0], axis=0)
    covariances_ = [cov[0] for cov in covariances_]
    return covariances_

class Orbits:
    """
    Orbits Class



    """
    def __init__(
            self,
            orbits,
            epochs,
            orbit_type="cartesian",
            orbit_units=CARTESIAN_UNITS,
            covariance=None,
            ids=None,
            H=None,
            G=None,
            orbit_class=None,
            additional_data=None
        ):
        """
        Class to store orbits and a variety of other useful attributes and
        data required to:
            propagate orbits
            generate ephemerides
            transform orbital elements

        """

        # Make sure that the given epoch(s) are an astropy time object
        if len(epochs) > 0:
            _checkTime(epochs, "epoch")

        # Make sure that each orbit has a an epoch
        assert len(epochs) == orbits.shape[0]
        self._epochs = epochs
        self.num_orbits = orbits.shape[0]

        # Make sure the passed orbits are one of the the supported types
        self._cartesian = None
        self._keplerian = None

        if orbit_type == "cartesian":
            if orbit_units != CARTESIAN_UNITS:
                orbits_ = _convertOrbitUnits(
                    orbits,
                    orbit_units,
                    CARTESIAN_UNITS
                )
            else:
                orbits_ = orbits.copy()

            self._cartesian = orbits_

        elif orbit_type == "keplerian":
            if orbit_units != KEPLERIAN_UNITS:
                orbits = _convertOrbitUnits(
                    orbits,
                    orbit_units,
                    KEPLERIAN_UNITS
                )
            else:
                orbits_ = orbits.copy()

            self._keplerian = orbits_

        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)

        self.orbit_type = orbit_type

        # If object IDs have been passed make sure each orbit has one
        if ids is not None:
            assert len(ids) == self.num_orbits
            self.ids = np.asarray(ids, dtype="<U60")
        else:
            self.ids = np.array(["{}".format(i) for i in range(self.num_orbits)], dtype="<U60")

        # If H magnitudes have been passed make sure each orbit has one
        if H is not None:
            assert len(H) == self.num_orbits
            self.H = np.asarray(H)
        else:
            self.H = None

        # If the slope parameter G has been passed make sure each orbit has one
        if G is not None:
            assert len(G) == self.num_orbits
            self.G = np.asarray(G)
        else:
            self.G = None

        # If covariances matrixes have been passed make sure each orbit has one
        self._cartesian_covariance = None
        self._keplerian_covariance = None
        if covariance is not None:
            assert len(covariance) == self.num_orbits
            if orbit_type == "cartesian":
                if orbit_units != CARTESIAN_UNITS:
                    covariance_ = _convertCovarianceUnits(
                        covariance,
                        orbit_units,
                        CARTESIAN_UNITS,
                    )
                else:
                    covariance_ = covariance

                self._cartesian_covariance = covariance_

            elif orbit_type == "keplerian":
                if orbit_units != KEPLERIAN_UNITS:
                    covariance_ = _convertCovarianceUnits(
                        covariance,
                        orbit_units,
                        KEPLERIAN_UNITS,
                    )
                else:
                    covariance_ = covariance

                self._keplerian_covariance = covariance_

            else:
                pass

        if orbit_class is not None:
            assert len(orbit_class) == self.num_orbits
            self.orbit_class = np.asarray(orbit_class)
        else:
            self.orbit_class = None

        if additional_data is not None:
            if isinstance(additional_data, pd.DataFrame):
                assert len(additional_data) == self.num_orbits
                self.additional_data = additional_data
            else:
                raise TypeError("additional_data must be a pandas DataFrame")
        else:
            self.additional_data = None

        return

    def __repr__(self):
        rep = (
            "Orbits: {}\n"
        )
        return rep.format(self.num_orbits)

    def __len__(self):
        return self.num_orbits

    def __getitem__(self, i):

        # Convert integer index to a slice so that array
        # based properties are not inappropriately reshaped
        if isinstance(i, int):
            i = slice(i, i+1)

        # Thrown an index error when the index is out of
        # range
        if i.start >= self.num_orbits:
            raise IndexError

        # Extract the relevant epochs
        _epochs = self.__dict__["_epochs"][i]

        # Extract the relevant orbital elements
        args = ()
        covariance = None
        if self.__dict__["_cartesian"] is not None and len(args) == 0:
            _cartesian = self.__dict__["_cartesian"][i]
            args = (_cartesian, _epochs)
            orbit_type = "cartesian"
            if self._cartesian_covariance is not None:
                covariance = self._cartesian_covariance[i]
        else:
            _cartesian = None

        if self.__dict__["_keplerian"] is not None and len(args) == 0:
            _keplerian = self.__dict__["_keplerian"][i]
            args = (_keplerian, _epochs)
            orbit_type = "_keplerian"
            if self._keplerian_covariance is not None:
                covariance = self._keplerian_covariance[i]
        else:
            _keplerian = None

        kwargs = {
            "orbit_type" : orbit_type,
            "covariance" : covariance,
        }
        for kwarg in ["ids", "H", "G", "orbit_class", "additional_data"]:
            if isinstance(self.__dict__[kwarg], np.ndarray):
                kwargs[kwarg] = self.__dict__[kwarg][i]
            elif isinstance(self.__dict__[kwarg], pd.DataFrame):
                kwargs[kwarg] = self.__dict__[kwarg].iloc[i]
                kwargs[kwarg].reset_index(
                    inplace=True,
                    drop=True
                )
            else:
                kwargs[kwarg] = self.__dict__[kwarg]

        orbit = Orbits(*args, **kwargs)
        orbit._cartesian = _cartesian
        orbit._keplerian = _keplerian
        orbit._cartesian_covariance = self._cartesian_covariance
        orbit._keplerian_covariance = self._keplerian_covariance
        return orbit

    def __eq__(self, other):

        eq = True
        while eq:
            if len(self) != len(other):
                return False

            if not np.all(np.isclose(self.cartesian, other.cartesian, rtol=1e-15, atol=1e-15)):
                return False

            if not np.all(np.isclose(self.epochs.tdb.mjd, other.epochs.tdb.mjd, rtol=1e-15, atol=1e-15)):
                return False

            if not np.all(self.ids == other.ids):
                return False

            break

        return eq

    @property
    def epochs(self):
        return self._epochs

    @property
    def cartesian(self):
        if not isinstance(self._cartesian, np.ndarray):
            logger.debug("Cartesian elements are not defined. Converting Keplerian elements to Cartesian.")
            self._cartesian = convertOrbitalElements(
                self._keplerian,
                "keplerian",
                "cartesian",
            )
        return self._cartesian

    @property
    def keplerian(self):
        if not isinstance(self._keplerian, np.ndarray):
            logger.debug("Keplerian elements are not defined. Converting Cartesian elements to Keplerian.")
            self._keplerian = convertOrbitalElements(
                self._cartesian,
                "cartesian",
                "keplerian",
            )
        return self._keplerian

    @property
    def cartesian_covariance(self):
        return self._cartesian_covariance

    @property
    def keplerian_covariance(self):
        return self._keplerian_covariance

    def split(self, chunk_size: int) -> list:
        """
        Split orbits into new orbit classes of size chunk_size.

        Parameters
        ----------
        chunk_size : int
            Size of each chunk of orbits.

        Returns
        -------
        list of Orbits
        """
        objs = []
        for chunk in range(0, self.num_orbits, chunk_size):
            objs.append(self[chunk:chunk+chunk_size])
        return objs

    def assignOrbitClasses(self):

        a = self.keplerian[:, 0]
        e = self.keplerian[:, 1]
        i = self.keplerian[:, 2]
        q = a * (1 - e)
        p = Q = a * (1 + e)

        classes = np.array(["AST" for i in range(len(self.keplerian))])

        classes_dict = {
            "AMO" : np.where((a > 1.0) & (q > 1.017) & (q < 1.3)),
            "MBA" : np.where((a > 1.0) & (q < 1.017)),
            "ATE" : np.where((a < 1.0) & (Q > 0.983)),
            "CEN" : np.where((a > 5.5) & (a < 30.3)),
            "IEO" : np.where((Q < 0.983))[0],
            "IMB" : np.where((a < 2.0) & (q > 1.666))[0],
            "MBA" : np.where((a > 2.0) & (a < 3.2) & (q > 1.666)),
            "MCA" : np.where((a < 3.2) & (q > 1.3) & (q < 1.666)),
            "OMB" : np.where((a > 3.2) & (a < 4.6)),
            "TJN" : np.where((a > 4.6) & (a < 5.5) & (e < 0.3)),
            "TNO" : np.where((a > 30.1)),
            "PAA" : np.where((e == 1)),
            "HYA" : np.where((e > 1)),
        }
        for c, v in classes_dict.items():
            classes[v] = c

        self.orbit_class = classes
        return

    @staticmethod
    def fromHorizons(obj_ids, t0):
        """
        Query Horizons for state vectors for each object ID at time t0.
        This is a convenience function and should not be used to query for state
        vectors for many objects or at many times.

        Parameters
        ----------
        obj_ids : `~numpy.ndarray` (N)
            Object IDs / designations recognizable by HORIZONS.
        t0 : `~astropy.core.time.Time` (1)
            Astropy time object at which to gather state vectors.

        Return
        ------
        `~thor.orbits.orbits.Orbits`
            THOR Orbits class
        """

        if len(t0) != 1:
            err = (
                "t0 should be a single time."
            )
            raise ValueError(err)

        horizons_vectors = getHorizonsVectors(
            obj_ids,
            t0,
            location="@sun",
            id_type="smallbody",
            aberrations="geometric"
        )

        orbits = Orbits(
            horizons_vectors[["x", "y", "z", "vx", "vy", "vz"]].values,
            t0 + np.zeros(len(obj_ids)),
            orbit_type="cartesian",
            orbit_units=CARTESIAN_UNITS,
            ids=horizons_vectors["targetname"].values,
            H=horizons_vectors["H"].values,
            G=horizons_vectors["G"].values,
        )
        return orbits

    @staticmethod
    def fromMPCOrbitCatalog(mpcorb):


        cols = ["a_au", "e", "i_deg", "ascNode_deg", "argPeri_deg", "meanAnom_deg"]
        additional_cols = mpcorb.columns[~mpcorb.columns.isin(cols + ["mjd_tt", "designation", "H_mag", "G"])]
        orbits = mpcorb[cols].values
        epochs = Time(
            mpcorb["mjd_tt"].values,
            scale="tt",
            format="mjd"
        )
        args = (orbits, epochs)

        kwargs = {
            "orbit_type" : "keplerian",
            "ids" : mpcorb["designation"].values,
            "H" : mpcorb["H_mag"].values,
            "G" : mpcorb["G"].values,
            "orbit_units" : KEPLERIAN_UNITS,
            "additional_data" : mpcorb[additional_cols]
        }
        return Orbits(*args, **kwargs)

    def to_df(
            self,
            include_units: bool = True,
            include_keplerian: bool = False,
            include_cartesian: bool = True,
        ) -> pd.DataFrame:
        """
        Convert orbits into a pandas dataframe.

        Returns
        -------
        dataframe : `~pandas.DataFrame`
            Dataframe containing orbits, epochs and IDs. If H, G, and covariances are defined then
            those are also added to the dataframe.
        """
        if self.num_orbits > 0:
            data = {
                "orbit_id" : self.ids,
                "mjd_tdb" : self.epochs.tdb.mjd
            }
        else:
            data = {
                "orbit_id" :[],
                "mjd_tdb" : [],
            }

        if include_units:

            units_index = ["--", "mjd [TDB]"]
            data["epoch"] = data["mjd_tdb"]
            data.pop("mjd_tdb")

        if include_cartesian:
            for i in range(6):
                data[CARTESIAN_COLS[i]] = self.cartesian[:, i]
            if include_units:
                orbit_units_str = []
                for unit in CARTESIAN_UNITS:
                    orbit_units_str.append(str(unit).lower())

                units_index += orbit_units_str

        if include_keplerian:
            for i in range(6):
                data[KEPLERIAN_COLS[i]] = self.keplerian[:, i]

            if include_units:
                orbit_units_str = []
                for unit in KEPLERIAN_UNITS:
                    if unit == u.dimensionless_unscaled:
                        orbit_units_str.append("--")
                    else:
                        orbit_units_str.append(str(unit).lower())
                units_index += orbit_units_str

        if self._cartesian_covariance is not None and include_cartesian:
            data["covariance"] = self.cartesian_covariance
            if include_units:
                units_index.append("--")

        if self._keplerian_covariance is not None and include_keplerian:
            data["covariance"] = self.keplerian_covariance
            if include_units:
                units_index.append("--")

        if self.H is not None:
            data["H"] = self.H
            if include_units:
                units_index.append("mag")

        if self.G is not None:
            data["G"] = self.G
            if include_units:
                units_index.append("--")

        if self.orbit_class is not None:
            data["orbit_class"] = self.orbit_class
            if include_units:
                units_index.append("--")

        dataframe = pd.DataFrame(data)
        if self.additional_data is not None:
            dataframe = dataframe.join(self.additional_data)
            if include_units:
                for col in self.additional_data.columns:
                    units_index.append("--")

        if include_units:
            dataframe.columns = pd.MultiIndex.from_arrays(
                [dataframe.columns, np.array(units_index)]
            )

        return dataframe

    @staticmethod
    def from_df(dataframe: pd.DataFrame):
        """
        Read orbits from a dataframe. Required columns are
        the epoch at which the orbits are defined ('mjd_tdb') and the 6 dimensional state.
        If the states are cartesian then the expected columns are ('x', 'y', 'z', 'vx', 'vy', 'vz'),
        if the states are keplerian then the expected columns are ("a", "e", "i", "raan", "argperi", "M").

        Parameters
        ----------
        dataframe : `~pandas.DataFrame`
            Dataframe containing either Cartesian or Keplerian orbits.

        Returns
        -------
        `~thor.orbits.orbits.Orbits`
            THOR Orbit class with the orbits read from the input dataframe.
        """
        # Extract the epochs from the given dataframe
        dataframe_ = dataframe.copy()

        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe_.columns = dataframe_.columns.droplevel(level=1)
            dataframe_.rename(
                columns={"epoch" : "mjd_tdb"},
                inplace=True
            )

        if len(dataframe_) > 0:
            epochs = Time(
                dataframe_["mjd_tdb"].values,
                format="mjd",
                scale="tdb"
            )
        else:
            epochs = np.array([])

        # If the dataframe's index is not sorted and increasing, reset it
        if not np.all(dataframe_.index.values == np.arange(0, len(dataframe_))):
            dataframe_.reset_index(
                inplace=True,
                drop=True
            )

        columns_required = ["orbit_id", "mjd_tdb"]
        cartesian = None
        keplerian = None
        args = ()
        if np.all(pd.Series(CARTESIAN_COLS).isin(dataframe_.columns)):
            columns_required += CARTESIAN_COLS
            cartesian = dataframe_[CARTESIAN_COLS].values
            if len(args) == 0:
                args = (cartesian, epochs)
                orbit_type = "cartesian"

        if  np.all(pd.Series(KEPLERIAN_COLS).isin(dataframe_.columns)):
            columns_required += KEPLERIAN_COLS
            keplerian = dataframe_[KEPLERIAN_COLS].values
            if len(args) == 0:
                args = (keplerian, epochs)
                orbit_type = "keplerian"

        kwargs = {
            "ids" : dataframe_["orbit_id"].values,
            "orbit_type" : orbit_type
        }

        columns_optional = ["covariance", "H", "G", "orbit_class"]
        for col in columns_optional:
            if col in dataframe_.columns:
                kwargs[col] = dataframe_[col].values

        columns = columns_required + columns_optional
        if len(dataframe_.columns[~dataframe_.columns.isin(columns)]) > 0:
            kwargs["additional_data"] = dataframe_[dataframe_.columns[~dataframe_.columns.isin(columns)]]

        orbits = Orbits(*args, **kwargs)
        orbits._keplerian = keplerian
        orbits._cartesian = cartesian
        #orbits._keplerian_covariance = keplerian_covariance
        #orbits._cartesian_covariance = cartesian_covariance
        return orbits

    def to_csv(
            self,
            file: str,
            include_cartesian: bool = True,
            include_keplerian: bool = False
        ):
        """
        Save orbits to a csv. Orbits are always saved with the
        units of each quantity to avoid ambiguity and confusion.

        Parameters
        ----------
        file : str
            Name or path of file including extension to which to save
            orbits.
        include_cartesian : bool
            Save Cartesian elements.
        include_keplerian : bool
            Save Keplerian elements.
        """
        df = self.to_df(
            include_units=True,
            include_cartesian=include_cartesian,
            include_keplerian=include_keplerian
        )
        df.to_csv(
            file,
            index=False,
            float_format="%.15e",
            encoding="utf-8"
        )
        return

    @staticmethod
    def from_csv(file: str):
        """
        Read orbits from a csv.

        Parameters
        ----------
        file : str
            Name or path of file including extension to which to read
            orbits.
        """
        df = pd.read_csv(
            file,
            index_col=None,
            header=[0, 1],
            converters={
                "cartesian_covariance" : lambda x: np.array(ast.literal_eval(','.join(x.replace('[ ', '[').split()))),
                "keplerian_covariance" : lambda x: np.array(ast.literal_eval(','.join(x.replace('[ ', '[').split())))
            },
            dtype={
                "orbit_id" : str,
                "test_orbit_id" : str,
            },
            float_precision="round_trip"
        )
        df.rename(
            columns={"epoch" : "mjd_tdb"},
            inplace=True
        )
        return Orbits.from_df(df)

    def to_hdf(
            self,
            file: str,
            include_cartesian: bool = True,
            include_keplerian: bool = False
        ):
        """
        Save orbits to a HDF5 file. Orbits are always saved with the
        units of each quantity to avoid ambiguity and confusion.

        Parameters
        ----------
        file : str
            Name or path of file including extension to which to save
            orbits.
        include_cartesian : bool
            Save Cartesian elements.
        include_keplerian : bool
            Save Keplerian elements.
        """
        df = self.to_df(
            include_units=True,
            include_cartesian=include_cartesian,
            include_keplerian=include_keplerian
        )
        df.to_hdf(
            file,
            key="data"
        )
        return

    @staticmethod
    def from_hdf(file: str):
        """
        Read orbits from a HDF5 file.

        Parameters
        ----------
        file : str
            Name or path of file including extension to which to read
            orbits.
        """
        df = pd.read_hdf(
            file,
            key="data"
        )
        return Orbits.from_df(df)