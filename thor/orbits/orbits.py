import numpy as np
import pandas as pd
from astropy.time import Time
from ..utils import _checkTime
from ..utils import getHorizonsVectors
from .kepler import convertOrbitalElements

CARTESIAN_COLS = ["x", "y", "z", "vx", "vy", "vz"]
KEPLERIAN_COLS = ["a", "e", "i", "raan", "argperi", "M"]

__all__ = ["Orbits"]

class Orbits:

    def __init__(
            self, 
            orbits, 
            epochs,
            orbit_type="cartesian",
            covariances=None,
            ids=None,
            H=None,
            G=None,
        ):
        # Make sure that the given epoch(s) are an astropy time object
        _checkTime(epochs, "epoch")

        # Make sure that each orbit has a an epoch
        assert len(epochs) == orbits.shape[0]
        self.epochs = epochs
        self.num_orbits = orbits.shape[0]

        # Make sure the passed orbits are one of the the supported types
        if orbit_type == "cartesian":
            self.cartesian = orbits
        elif orbit_type == "keplerian":
            self.keplerian = orbits
            self.cartesian = convertOrbitalElements(
                orbits, 
                "keplerian", 
                "cartesian"
            )
        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)
        self.orbit_type = "cartesian"

        # If object IDs have been passed make sure each orbit has one
        if ids is not None:
            assert len(ids) == self.num_orbits
            self.ids = np.asarray(ids)
        else:
            self.ids = np.arange(0, self.num_orbits, 1)

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
        if covariances is not None:
            assert len(covariances) == self.num_orbits
            self.covariances = covariances
        else:
            self.covariances = None

        return

    def __repr__(self):
        rep = (
            "Orbits: {}\n"
            "Type: {}\n"
        )
        return rep.format(self.num_orbits, self.orbit_type)

    def __len__(self):
        return self.num_orbits

    def __getitem__(self, i):
        args = []
        for arg in ["cartesian", "epochs"]:
            args.append(self.__dict__[arg][i])
        
        kwargs = {}
        for kwarg in ["orbit_type", "ids", "H", "G", "covariances"]:
            if type(self.__dict__[kwarg]) == np.ndarray:
                kwargs[kwarg] = self.__dict__[kwarg][i]
            else:
                kwargs[kwarg] = self.__dict__[kwarg]

        return Orbits(*args, **kwargs)

    def split(self, chunk_size):
        objs = []
        for chunk in range(0, self.num_orbits, chunk_size):
            args = []
            for arg in ["cartesian", "keplerian", "epochs"]:
                if arg in self.__dict__.keys():
                    args.append(self.__dict__[arg][chunk:chunk + chunk_size].copy())
            
            kwargs = {}
            for kwarg in ["orbit_type", "ids", "H", "G", "covariances"]:
                if type(self.__dict__[kwarg]) == np.ndarray:
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
        vectors for many objects.

        Parameters
        ----------
        obj_ids : `~numpy.ndarray` (N)
            Object IDs / designations recognizable by HORIZONS. 
        times : `~astropy.core.time.Time` (1)
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
            ids=horizons_vectors["targetname"].values,
            H=horizons_vectors["H"].values,
            G=horizons_vectors["G"].values,
        )
        return orbits

    @staticmethod
    def fromdf(dataframe, orbit_type="cartesian"):
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
        epochs = Time(
            dataframe["mjd_tdb"].values,
            format="mjd",
            scale="tdb"
        )

        # Assert orbit type is one of two types otherwise
        # raise a value error
        if orbit_type == "cartesian":
            states = dataframe[CARTESIAN_COLS].values
        
        elif orbit_type == "keplerian":
            states = dataframe[KEPLERIAN_COLS].values

        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)

        args = (states, epochs)
        kwargs = {
            "orbit_type" : orbit_type
        }

        if "orbit_id" in dataframe.columns:
            kwargs["ids"] = dataframe["orbit_id"].values

        if "covariance" in dataframe.columns:
            kwargs["covariances"] = dataframe["covariance"].values

        for c in ["H", "G"]:
            if c in dataframe.columns:
                kwargs[c] = dataframe[c].values

        return Orbits(*args, **kwargs)
    
    def todf(self):
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

        if self.G is not None:
            data["G"] = self.G

        if self.covariances is not None:
            data["covariances"] = self.covariances

        return pd.DataFrame(data)