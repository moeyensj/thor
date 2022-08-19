import warnings
import numpy as np
import pandas as pd
from typing import Optional

from ..utils.indexable import Indexable
from ..coordinates.coordinates import Coordinates
from ..coordinates.members import CoordinateMembers

__all__ = [
    "generate_ephemeris",
    "Ephemeris"
]

def generate_ephemeris(
        orbits,
        observers,
        backend="PYOORB",
        backend_kwargs={},
        chunk_size=1,
        num_jobs=1,
        parallel_backend="mp"
    ):
    """
    Generate ephemeris for the orbits and the given observatories.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits for which to generate ephemeris. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be
        expressed in keplerian, cometary or cartesian elements.
    observers : dict
        A dictionary with observatory codes as keys and observation_times (`~astropy.time.core.Time`) as values.
        Or a data frame with observatory codes, observation times (in UTC), and the observer's heliocentric ecliptic state.
        The expected data frame columns are obs_x, obs_y, obs_y and optionally the velocity columns obs_vx, obs_vy, obs_vz.
        If no velocities are not correctly given, then sky-plane velocities will all be zero.
        (See: `~thor.observers.getObserverState`)
    backend : {'PYOORB', 'FINDORB'}, optional
        Which backend to use.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        backend.
    chunk_size : int, optional
        Number of orbits to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
        module ('mp').

    Returns
    -------
    ephemeris : `~pandas.DataFrame` (N x M, 21) or (N x M, 18)
        A DataFrame containing the generated ephemeris.
    """
    from ..backend import Backend
    if backend == "PYOORB":
        from ..backend import PYOORB
        backend = PYOORB(**backend_kwargs)

    elif backend == "FINDORB":
        from ..backend import FINDORB
        backend = FINDORB(**backend_kwargs)

    elif isinstance(backend, Backend):
        backend = backend

        if len(backend_kwargs) > 0:
            warnings.warn("backend_kwargs will be ignored since a instantiated backend class has been given.")

    else:
        err = (
            "backend should be one of 'PYOORB', 'FINDORB' or an instantiated Backend class"
        )
        raise ValueError(err)

    ephemeris = backend.generate_ephemeris(
        orbits,
        observers,
        chunk_size=chunk_size,
        num_jobs=num_jobs,
        parallel_backend=parallel_backend
    )
    return ephemeris


class Ephemeris(CoordinateMembers):


    def __init__(
            self,
            coordinates: Coordinates,
            orbit_ids: np.ndarray,
            object_ids: Optional[np.ndarray] = None,
            light_time: Optional[np.ndarray] = None,
        ):
        CoordinateMembers.__init__(
            self,
            coordinates=coordinates,
            cartesian=True,
            spherical=True,
            keplerian=False,
            cometary=False
        )

        if object_ids is not None:
            self._object_ids = object_ids
        else:
            self._object_ids = np.array(["None" for i in range(len(coordinates))])

        if isinstance(light_time, np.ndarray):
            self._light_time = light_time
        else:
            self._light_time = None

        self._orbit_ids = orbit_ids
        Indexable.__init__(self, index="orbit_ids")
        return

    @property
    def orbit_ids(self):
        return self._orbit_ids

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def light_time(self):
        return self._light_time

    def to_df(self,
            time_scale: str = "utc",
            coordinate_type: Optional[str] = None,
            sigmas: bool = False,
            covariances: bool = False,
        ) -> pd.DataFrame:
        """
        Represent ephemeris as a `~pandas.DataFrame`.

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
            Pandas DataFrame containing orbits.
        """
        df = CoordinateMembers.to_df(self,
            time_scale=time_scale,
            coordinate_type=coordinate_type,
            sigmas=sigmas,
            covariances=covariances,
        )
        df.insert(0, "orbit_id", self.orbit_ids)
        df.insert(1, "object_id", self.object_ids)
        df.insert(len(df.columns), "light_time", self.light_time)
        return df

