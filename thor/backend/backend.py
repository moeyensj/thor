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

from ..orbit import TestOrbit
from ..utils import Timeout
from ..utils import _initWorker
from ..utils import _checkParallel

logger = logging.getLogger(__name__)

__all__ = [
    "Backend"
]

TIMEOUT = 30

def propagation_worker(orbits, t1, backend):
    with Timeout(seconds=TIMEOUT):
        try:
            propagated = backend._propagate_orbits(orbits, t1)
        except TimeoutError:
            logger.critical("Propagation timed out on orbit IDs (showing first 5): {}".format(orbits.ids[:5]))
            propagated = pd.DataFrame()
    return propagated

def ephemeris_worker(orbits, observers, backend):
    with Timeout(seconds=TIMEOUT):
        try:
            ephemeris = backend._generate_ephemeris(orbits, observers)
        except TimeoutError:
            logger.critical("Ephemeris generation timed out on orbit IDs (showing first 5): {}".format(orbits.ids[:5]))
            ephemeris = pd.DataFrame()
    return ephemeris

def orbitDetermination_worker(observations, backend):
    with Timeout(seconds=TIMEOUT):
        try:
            orbits = backend._orbit_determination(observations)
        except TimeoutError:
            logger.critical("Orbit determination timed out on observations (showing first 5): {}".format(observations["obs_id"].values[:5]))
            orbits = pd.DataFrame()
    return orbits

def projectEphemeris_worker(ephemeris, test_orbit_ephemeris):

    assert len(ephemeris["mjd_utc"].unique()) == 1
    assert len(test_orbit_ephemeris["mjd_utc"].unique()) == 1
    assert ephemeris["mjd_utc"].unique()[0] == test_orbit_ephemeris["mjd_utc"].unique()[0]
    observation_time = ephemeris["mjd_utc"].unique()[0]

    # Create test orbit with state of orbit at visit time
    test_orbit = TestOrbit(
        test_orbit_ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values[0],
        observation_time
    )

    # Prepare rotation matrices
    test_orbit.prepare()

    # Apply rotation matrices and transform observations into the orbit's
    # frame of motion.
    test_orbit.applyToEphemeris(ephemeris)

    return ephemeris

class Backend:

    def __init__(self, name="Backend", **kwargs):
        self.__dict__.update(kwargs)
        self.name = name
        self.is_setup = False
        return

    def setup(self):
        return

    def _propagate_orbits(self, orbits, t1):
        """
        Propagate orbits from t0 to t1.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.

        """
        err = (
            "This backend does not have orbit propagation implemented."
        )
        raise NotImplementedError(err)

    def propagate_orbits(
            self,
            orbits,
            t1,
            chunk_size=100,
            num_jobs=1,
            parallel_backend="mp"
        ):
        """
        Propagate each orbit in orbits to each time in t1.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits to propagate.
        t1 : `~astropy.time.core.Time`
            Times to which to propagate each orbit.
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
            orbits_split = orbits.split(chunk_size)
            t1_duplicated = [copy.deepcopy(t1) for i in range(len(orbits_split))]
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
                for o, t, b in zip(orbits_split, t1_duplicated, backend_duplicated):
                    p.append(propagation_worker_ray.remote(o, t, b))
                propagated_dfs = ray.get(p)

            else: # parallel_backend == "mp"
                p = mp.Pool(
                    processes=num_workers,
                    initializer=_initWorker,
                )

                propagated_dfs = p.starmap(
                    propagation_worker,
                    zip(
                        orbits_split,
                        t1_duplicated,
                        backend_duplicated,
                    )
                )
                p.close()

            propagated = pd.concat(propagated_dfs)
            propagated.reset_index(
                drop=True,
                inplace=True
            )
        else:
            propagated = self._propagate_orbits(
                orbits,
                t1
            )

        return propagated

    def _generate_ephemeris(self, orbits, observers):
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
            orbits,
            observers,
            chunk_size=100,
            num_jobs=1,
            parallel_backend="mp"
        ):
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits for which to generate ephemerides.
        observers : dict or `~pandas.DataFrame`
            A dictionary with observatory codes as keys and observation_times (`~astropy.time.core.Time`) as values.
        chunk_size : int, optional
            Number of orbits to send to each job.
        num_jobs : int, optional
            Number of jobs to launch.
        parallel_backend : str, optional
            Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
            module ('mp').

        Returns
        -------
        ephemeris : `~pandas.DataFrame`
            Ephemerides with at least the following columns:
                orbit_id : Input orbit ID
                observatory_code : Observatory's MPC code.
                mjd_utc : Observation time in MJD UTC.
                RA : Right Ascension in decimal degrees.
                Dec : Declination in decimal degrees.
        """
        parallel, num_workers = _checkParallel(num_jobs, parallel_backend)
        if parallel:
            orbits_split = orbits.split(chunk_size)
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
                ephemeris_dfs = ray.get(p)

            else: # parallel_backend == "mp"
                p = mp.Pool(
                    processes=num_workers,
                    initializer=_initWorker,
                )

                ephemeris_dfs = p.starmap(
                    ephemeris_worker,
                    zip(
                        orbits_split,
                        observers_duplicated,
                        backend_duplicated,
                    )
                )
                p.close()

            ephemeris = pd.concat(ephemeris_dfs)
            ephemeris.reset_index(
                drop=True,
                inplace=True
            )
        else:
            ephemeris = self._generate_ephemeris(
                orbits,
                observers
            )

        ephemeris.sort_values(
            by=["orbit_id", "observatory_code", "mjd_utc"],
            inplace=True,
            ignore_index=True
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