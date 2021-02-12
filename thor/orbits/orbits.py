import ast
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from ..utils import _checkTime
from ..utils import getHorizonsVectors
from .kepler import convertOrbitalElements

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
CARTESIAN_UNITS = [u.au, u.au, u.au, u.au / u.d,  u.au / u.d,  u.au / u.d]
KEPLERIAN_COLS = ["a", "e", "i", "raan", "argperi", "M0"]
KEPLERIAN_UNITS = [u.au, u.dimensionless_unscaled, u.deg, u.deg, u.deg, u.deg]

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
            num_obs=None,
        ):
        """
        Class to store orbits and a variety of other useful attributes and 
        data required to:
            propagate orbits
            generate ephemerides
            transform orbital elements

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Heliocentric ecliptic elements in one of two formats:
            - cartesian : 
                x-, y-, z- positions in au (x, y, z)
                x-, y-, z- velocities in au per day (vx, vy, vz)
            - keplerian : 
                a : semi major axis in au
                e : eccentricity
                i : inclination in degrees

        epochs : `~astropy.time.core.Time` (N)
            Time at which each orbit is defined.
        orbit_type : {'cartesian', 'keplerian`}
            The type of the input orbits. If keplerian, this class
            will convert the keplerian elements to cartesian for use in 
            THOR. THOR uses only cartesian elements but the user may provide
            alternate orbital reference frames for elements. 
        covariance : list of `~numpy.ndarray (6, 6)`, optional
            The covariance matrices for each orbital elements. Covariances are currently
            not used by THOR but generated during orbit determination.



            


        """

        # Make sure that the given epoch(s) are an astropy time object
        _checkTime(epochs, "epoch")

        # Make sure that each orbit has a an epoch
        assert len(epochs) == orbits.shape[0]
        self.epochs = epochs
        self.num_orbits = orbits.shape[0]

        # Make sure the passed orbits are one of the the supported types
        self.cartesian = None
        self.keplerian = None
        if orbit_type == "cartesian":
            if orbit_units != CARTESIAN_UNITS:
                orbits_ = _convertOrbitUnits(
                    orbits,
                    orbit_units,
                    CARTESIAN_UNITS
                )
            else:
                orbits_ = orbits.copy()
            
            self.cartesian = orbits_
            self.orbit_units = CARTESIAN_UNITS
        elif orbit_type == "keplerian":
            if orbit_units != KEPLERIAN_UNITS:
                orbits = _convertOrbitUnits(
                    orbits,
                    orbit_units,
                    KEPLERIAN_UNITS
                )
            else:
                orbits_ = orbits.copy()
            
            self.keplerian = orbits_
            self.orbit_units = KEPLERIAN_UNITS
        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)

        self.orbit_type = orbit_type
        if self.cartesian is None:
            self.cartesian = convertOrbitalElements(
                self.keplerian, 
                "keplerian", 
                "cartesian"
            )
            #self.orbit_type = "cartesian"


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
        if covariance is not None:
            assert len(covariance) == self.num_orbits
            if self.orbit_type == "cartesian" and orbit_units != CARTESIAN_UNITS:
                covariance_ = _convertCovarianceUnits(
                    covariance, 
                    orbit_units, 
                    CARTESIAN_UNITS,
                )
            elif self.orbit_type == "keplerian" and orbit_units != KEPLERIAN_UNITS:
                covariance_ = _convertCovarianceUnits(
                    covariance, 
                    orbit_units, 
                    KEPLERIAN_UNITS,
                )
            else:
                covariance_ = covariance

            self.covariance = covariance_
            
        else:
            self.covariance = None
            
        if num_obs is not None:
            assert len(num_obs) == self.num_orbits
            self.num_obs = num_obs
        else:
            self.num_obs = None

        return

    def __repr__(self):
        rep = (
            "Orbits: {}\n"
            "Type: {}\n"
            "Units: {}\n"
        )
        return rep.format(self.num_orbits, self.orbit_type, ", ".join([str(i).lower() for i in self.orbit_units]))

    def __len__(self):
        return self.num_orbits

    def __getitem__(self, i):
        args = []
        for arg in ["cartesian", "epochs"]:
            args.append(self.__dict__[arg][i])
        
        kwargs = {}
        for kwarg in ["orbit_type", "ids", "H", "G", "covariance"]:
            if isinstance(self.__dict__[kwarg], np.ndarray):
                kwargs[kwarg] = self.__dict__[kwarg][i]
            else:
                kwargs[kwarg] = self.__dict__[kwarg]

        return Orbits(*args, **kwargs)

    def split(self, chunk_size):
        objs = []
        for chunk in range(0, self.num_orbits, chunk_size):
            args = []
            for arg in ["cartesian", "epochs"]:
                if arg in self.__dict__.keys():
                    args.append(self.__dict__[arg][chunk:chunk + chunk_size].copy())
            
            kwargs = {}
            for kwarg in ["orbit_type", "ids", "H", "G", "covariance"]:
                if isinstance(self.__dict__[kwarg], np.ndarray):
                    kwargs[kwarg] = self.__dict__[kwarg][chunk:chunk + chunk_size].copy()
                else:
                    kwargs[kwarg] = self.__dict__[kwarg]
            
            objs.append(Orbits(*args, **kwargs))
        
        return objs

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

        orbits = mpcorb[["a_au", "e", "i_deg", "ascNode_deg", "argPeri_deg", "meanAnom_deg"]].values
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
        }
        return Orbits(*args, **kwargs)

    def to_df(self, include_units=True):
        """
        Convert orbits into a pandas dataframe.

        Returns
        -------
        dataframe : `~pandas.DataFrame`
            Dataframe containing orbits, epochs and IDs. If H, G, and covariances are defined then 
            those are also added to the dataframe.
        """
        data = {
            "orbit_id" : self.ids,
            "mjd_tdb" : self.epochs.tdb.mjd
        }

        if include_units:
            orbit_units_str = []
            for unit in self.orbit_units:
                if unit == u.dimensionless_unscaled:
                    orbit_units_str.append("--")
                else:
                    orbit_units_str.append(str(unit).lower())

            units_index = ["--", "mjd [TDB]"] + orbit_units_str

        if self.orbit_type == "cartesian":
            for i in range(6):
                data[CARTESIAN_COLS[i]] = self.cartesian[:, i]
        elif self.orbit_type == "keplerian":
            for i in range(6):
                data[KEPLERIAN_COLS[i]] = self.keplerian[:, i]
        else:
            pass
        
        if self.H is not None:
            data["H"] = self.H
            if include_units:
                units_index.append("mag")

        if self.G is not None:
            data["G"] = self.G
            if include_units:
                units_index.append("--")

        if self.covariance is not None:
            data["covariance"] = self.covariance
            if include_units:
                units_index.append("--")
            
        if self.num_obs is not None:
            data["num_obs"] = self.num_obs
            if include_units:
                units_index.append("--")

        dataframe = pd.DataFrame(data)
        if include_units:
            dataframe.columns = pd.MultiIndex.from_arrays(
                [dataframe.columns, np.array(units_index)]
            )
        return dataframe

    @staticmethod
    def from_df(dataframe, orbit_type="cartesian"):
        """
        Read orbits from a dataframe. Required columns are 
        the epoch at which the orbits are defined ('mjd_tdb') and the 6 dimensional state. 
        If the states are cartesian then the expected columns are ('x', 'y', 'z', 'vx', 'vy', 'vz'), 
        if the states are keplerian then the expected columns are ("a", "e", "i", "raan", "argperi", "M").

        Parameters
        ----------
        dataframe : `~pandas.DataFrame`
            Dataframe containing either cartesian or Keplerian orbits. 
        orbit_type : str, optional
            The 


        Returns
        -------
        `~thor.orbits.orbits.Orbits`
            THOR Orbit class with the orbits read from the input dataframe.
        """
        # Extract the epochs from the given dataframe
        dataframe_ = dataframe.copy()
        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe_.columns = dataframe_.columns.droplevel(level=1)

        epochs = Time(
            dataframe_["epoch"].values,
            format="mjd",
            scale="tdb"
        )

        # Assert orbit type is one of two types otherwise
        # raise a value error
        if orbit_type == "cartesian":
            states = dataframe_[CARTESIAN_COLS].values
        
        elif orbit_type == "keplerian":
            states = dataframe_[KEPLERIAN_COLS].values

        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)

        args = (states, epochs)
        kwargs = {
            "orbit_type" : orbit_type
        }

        if "orbit_id" in dataframe_.columns:
            kwargs["ids"] = dataframe_["orbit_id"].values

        if "covariance" in dataframe_.columns:
            kwargs["covariance"] = dataframe_["covariance"].values
            
        if "num_obs" in dataframe_.columns:
            kwargs["num_obs"] = dataframe_["num_obs"].values

        for c in ["H", "G"]:
            if c in dataframe_.columns:
                kwargs[c] = dataframe_[c].values

        return Orbits(*args, **kwargs)
    
    def to_csv(self, file):
        
        df = self.to_df(
            include_units=True
        )
        with np.printoptions(precision=17):
            df.to_csv(
                file,
            )
        return 

    @staticmethod
    def from_csv(file):

        df = pd.read_csv(
            file,
            index_col=0, 
            header=[0, 1], 
            converters={
                "covariance" : lambda x: np.array(ast.literal_eval(','.join(x.replace('[ ', '[').split())))
            }
        )
        return Orbits.from_df(df)