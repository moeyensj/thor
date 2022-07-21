import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import logging
import pandas as pd
import multiprocessing as mp
from astropy.time import Time

from ..orbits.orbits import Orbits
from ..orbits.classes import Ephemeris
from ..observers.observers import Observers
from ..utils.indexable import concatenate
from ..utils import Timeout
from ..utils import _initWorker
from ..utils import _checkParallel

logger = logging.getLogger(__name__)

__all__ = [
    "Backend"
]

TIMEOUT = 30

def propagation_worker(
        orbits: Orbits,
        times: Time,
        backend: "Backend"
    ) -> Orbits:
    with Timeout(seconds=TIMEOUT):
        try:
            propagated = backend._propagate_orbits(orbits, times)
        except TimeoutError:
            logger.critical("Propagation timed out on orbit IDs (showing first 5): {}".format(orbits.ids[:5]))
            propagated = pd.DataFrame()
    return propagated

def ephemeris_worker(
        orbits: Orbits,
        observers: Observers,
        backend: "Backend"
    ) -> Ephemeris:
    with Timeout(seconds=TIMEOUT):
        try:
            ephemeris = backend._generate_ephemeris(orbits, observers)
        except TimeoutError:
            logger.critical("Ephemeris generation timed out on orbit IDs (showing first 5): {}".format(orbits.ids[:5]))
            ephemeris = pd.DataFrame()
    return ephemeris

def orbit_determination_worker(observations, backend: "Backend"):
    with Timeout(seconds=TIMEOUT):
        try:
            orbits = backend._orbit_determination(observations)
        except TimeoutError:
            logger.critical("Orbit determination timed out on observations (showing first 5): {}".format(observations["obs_id"].values[:5]))
            orbits = pd.DataFrame()
    return orbits

class Backend:

    def __init__(self, name: str = "Backend", **kwargs):
        self.__dict__.update(kwargs)
        self.name = name
        self.is_setup = False
        return

    def setup(self):
        return

    def _propagate_orbits(self,
            orbits: Orbits,
            times: Time
        ) -> Orbits:
        """
        Propagate orbits from t0 to times.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.

        """
        err = (
            "This backend does not have orbit propagation implemented."
        )
        raise NotImplementedError(err)

    def propagate_orbits(
            self,
            orbits: Orbits,
            times: Time,
            chunk_size: int = 100,
            num_jobs: int = 1,
            parallel_backend: str = "mp"
        ) -> Orbits:
        """
        Propagate each orbit in orbits to each time in times.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits` (N)
            Orbits to propagate.
        times : `~astropy.time.core.Time` (M)
            Times to which to propagate orbits.
        chunk_size : int, optional
            Number of orbits to send to each job.
        num_jobs : int, optional
            Number of jobs to launch.
        parallel_backend : str, optional
            Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
            module ('mp').

        Returns
        -------
        propagated : `~pandas.DataFrame`
            Propagated orbits with at least the following columns:
                orbit_id : Input orbit ID.
                mjd_tdb : Time at which state is defined in MJD TDB.
                x, y, z, vx, vy, vz : Orbit as cartesian state vector with units
                of au and au per day.
        """
        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if parallel:
            orbits_split = list(orbits.yield_chunks(chunk_size))
            times_duplicated = [copy.deepcopy(times) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]


            if parallel_backend == "ray":
                import ray
                if not ray.is_initialized():
                    ray.init(address="auto")

                propagation_worker_ray = ray.remote(propagation_worker)
                propagation_worker_ray.options(
                    num_returns=1,
                    num_cpus=1
                )

                p = []
                for o, t, b in zip(orbits_split, times_duplicated, backend_duplicated):
                    p.append(propagation_worker_ray.remote(o, t, b))
                propagated_list = ray.get(p)

            else: # parallel_backend == "mp"
                p = mp.Pool(
                    processes=num_workers,
                    initializer=_initWorker,
                )

                propagated_list = p.starmap(
                    propagation_worker,
                    zip(
                        orbits_split,
                        times_duplicated,
                        backend_duplicated,
                    )
                )
                p.close()

            propagated = concatenate(propagated_list)
        else:
            propagated = self._propagate_orbits(
                orbits,
                times
            )

        propagated.sort_values(
            by=["orbit_ids", "times"],
            ascending=[True, True],
            inplace=True
        )
        return propagated

    def _generate_ephemeris(
            self,
            orbits: Orbits,
            observers: Observers
        ) -> Ephemeris:
        """
        Generate ephemerides for the given orbits as observed by
        the observers.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.

        """
        err = (
            "This backend does not have ephemeris generation implemented."
        )
        raise NotImplementedError(err)

    def generate_ephemeris(
            self,
            orbits: Orbits,
            observers: Observers,
            chunk_size: int = 100,
            num_jobs: int = 1,
            parallel_backend: str = "mp"
        ) -> Ephemeris:
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits` (N)
            Orbits for which to generate ephemerides.
        observers : `~thor.observers.observers.Observers` (M)
            Observers for which to generate the ephemerides of each
            orbit.
        chunk_size : int, optional
            Number of orbits to send to each job.
        num_jobs : int, optional
            Number of jobs to launch.
        parallel_backend : str, optional
            Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
            module ('mp').

        Returns
        -------
        ephemeris : `~thor.orbits.classes.Ephemeris` (N * M)
            Predicted ephemerides for each orbit observed by each
            observer.
        """
        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if parallel:
            orbits_split = list(orbits.yield_chunks(chunk_size))
            observers_duplicated = [copy.deepcopy(observers) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]

            if parallel_backend == "ray":
                import ray
                if not ray.is_initialized():
                    ray.init(address="auto")

                ephemeris_worker_ray = ray.remote(ephemeris_worker)
                ephemeris_worker_ray.options(
                    num_returns=1,
                    num_cpus=1
                )

                p = []
                for o, t, b in zip(orbits_split, observers_duplicated, backend_duplicated):
                    p.append(ephemeris_worker_ray.remote(o, t, b))
                ephemeris_list = ray.get(p)

            else: # parallel_backend == "mp"
                p = mp.Pool(
                    processes=num_workers,
                    initializer=_initWorker,
                )

                ephemeris_list = p.starmap(
                    ephemeris_worker,
                    zip(
                        orbits_split,
                        observers_duplicated,
                        backend_duplicated,
                    )
                )
                p.close()

            ephemeris = concatenate(ephemeris_list)

        else:
            ephemeris = self._generate_ephemeris(
                orbits,
                observers
            )

        ephemeris.sort_values(
            by=["orbit_ids", "origin", "times"],
            inplace=True,
        )
        return ephemeris

    def orbit_determination(self):
        err = (
            "This backend does not have orbit determination implemented."
        )
        raise NotImplementedError(err)

    def orbit_determination(
            self,
            observations,
            chunk_size=10,
            num_jobs=1,
            parallel_backend="mp"
        ):
        """
        Run orbit determination on the input observations. These observations
        must at least contain the following columns:

        obj_id : Object ID
        mjd_utc : Observation time in MJD UTC.
        RA_deg : Topocentric Right Ascension in decimal degrees.
        Dec_deg : Topocentric Declination in decimal degrees.
        sigma_RA_deg : 1-sigma uncertainty in RA.
        sigma_Dec_deg : 1-sigma uncertainty in Dec.
        observatory_code : MPC observatory code.

        Parameters
        ----------
        num_jobs : int, optional
            Number of jobs to launch.
        parallel_backend : str, optional
            Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
            module ('mp').
        """
        unique_objs = observations["obj_id"].unique()
        observations_split = [observations[observations["obj_id"].isin(unique_objs[i:i+chunk_size])].copy() for i in range(0, len(unique_objs), chunk_size)]
        backend_duplicated = [copy.deepcopy(self) for i in range(len(observations_split))]

        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if parallel_backend == "ray":
            import ray
            if not ray.is_initialized():
                ray.init(address="auto")

            orbit_determination_worker_ray = ray.remote(orbit_determination_worker)
            orbit_determination_worker_ray = orbit_determination_worker_ray.options(
                num_returns=1,
                num_cpus=1
            )

            od = []
            for o,  b in zip(observations_split, backend_duplicated):
                od.append(orbit_determination_worker_ray.remote(o, b))
            od_orbits_dfs = ray.get(od)

        else: # parallel_backend == "mp"
            p = mp.Pool(
                processes=num_workers,
                initializer=_initWorker,
            )

            od_orbits_dfs = p.starmap(
                orbit_determination_worker,
                zip(
                    observations_split,
                    backend_duplicated,
                )
            )
            p.close()

        od_orbits = pd.concat(od_orbits_dfs, ignore_index=True)
        return od_orbits

    def _get_observer_state(self, observers, origin="heliocenter"):
        err = (
            "This backend does not have observer state calculations implemented."
        )
        raise NotImplementedError(err)