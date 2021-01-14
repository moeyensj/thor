import os
import copy
import signal
import pandas as pd
import multiprocessing as mp

from ..config import Config
from ..orbit import TestOrbit

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

__all__ = [
    "_init_worker",
    "Backend"
]

def _init_worker():
    """
    Tell multiprocessing worker to ignore signals, will only
    listen to parent process. 
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return

def propagation_worker(orbits, t1, backend):
    propagated = backend._propagateOrbits(orbits, t1)
    return propagated

def ephemeris_worker(orbits, observers, backend):
    ephemeris = backend._generateEphemeris(orbits, observers)
    return ephemeris

def orbitdetermination_worker(observations, backend):
    orbits = backend._orbitDetermination(observations)
    return orbits

def projectephemeris_worker(ephemeris, test_orbit_ephemeris):
    
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
    test_orbit.prepare(verbose=False)

    # Apply rotation matrices and transform observations into the orbit's
    # frame of motion. 
    test_orbit.applyToEphemeris(ephemeris, verbose=False)
    
    return ephemeris

if USE_RAY:
    import ray
    propagation_worker = ray.remote(propagation_worker)
    ephemeris_worker = ray.remote(ephemeris_worker)
    orbitdetermination_worker = ray.remote(orbitdetermination_worker)
    projectephemeris_worker = ray.remote(projectephemeris_worker)

class Backend:
    
    def __init__(self, name="Backend", **kwargs):
        self.__dict__.update(kwargs)
        self.name = name
        self.is_setup = False
        return  
    
    def setup(self):
        return
    
    def _propagateOrbits(self, orbits, t1):
        """
        Propagate orbits from t0 to t1. 

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        
        """
        err = (
            "This backend does not have orbit propagation implemented."
        )
        raise NotImplementedError(err)

    def propagateOrbits(self, orbits, t1, threads=NUM_THREADS, chunk_size=100):
        """
        Propagate each orbit in orbits to each time in t1.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits to propagate.
        t1 : `~astropy.time.core.Time`
            Times to which to propagate each orbit.
        threads : int, optional
            Number of processes to launch.
        chunk_size : int, optional
            Number of orbits to send to each process.

        Returns
        -------
        propagated : `~pandas.DataFrame`
            Propagated orbits with at least the following columns:
                orbit_id : Input orbit ID.
                epoch_mjd_tdb : Time at which state is defined in MJD TDB.
                x, y, z, vx, vy, vz : Orbit as cartesian state vector with units 
                of au and au per day. 
        """
        if threads > 1:
            orbits_split = orbits.split(chunk_size)
            t1_duplicated = [copy.deepcopy(t1) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]

            if USE_RAY:
                shutdown = False
                if not ray.is_initialized():
                    ray.init(num_cpus=threads)
                    shutdown = True
            
                p = []
                for o, t, b in zip(orbits_split, t1_duplicated, backend_duplicated):
                    p.append(propagation_worker.remote(o, t, b))
                propagated_dfs = ray.get(p)

                if shutdown:
                    ray.shutdown()
            else:
                p = mp.Pool(
                    processes=threads,
                    initializer=_init_worker,
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
            propagated = self._propagateOrbits(
                orbits,
                t1
            )
            
        return propagated
    
    def _generateEphemeris(self, orbits, observers):
        """
        Generate ephemerides for the given orbits as observed by 
        the observers.

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        
        """
        err = (
            "This backend does not have ephemeris generation implemented."
        )
        raise NotImplementedError(err)

    def generateEphemeris(self, orbits, observers, test_orbit=None, threads=NUM_THREADS, chunk_size=100):
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits for which to generate ephemerides.
        observers : dict or `~pandas.DataFrame`
            A dictionary with observatory codes as keys and observation_times (`~astropy.time.core.Time`) as values. 
        test_orbit : `~thor.orbits.orbits.Orbits`
            Test orbit to use to generate projected coordinates.
        threads : int, optional
            Number of processes to launch.
        chunk_size : int, optional
            Number of orbits to send to each process.

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
        shutdown = False
        if threads > 1:
            orbits_split = orbits.split(chunk_size)
            observers_duplicated = [copy.deepcopy(observers) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]

            if USE_RAY:
                shutdown = False
                if not ray.is_initialized():
                    ray.init(num_cpus=threads)
                    shutdown = True
            
                p = []
                for o, t, b in zip(orbits_split, observers_duplicated, backend_duplicated):
                    p.append(ephemeris_worker.remote(o, t, b))
                ephemeris_dfs = ray.get(p)

            else:
                p = mp.Pool(
                    processes=threads,
                    initializer=_init_worker,
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
            ephemeris = self._generateEphemeris(
                orbits,
                observers
            )

        if test_orbit is not None:

            test_orbit_ephemeris = self._generateEphemeris(
                test_orbit,
                observers
            )
            ephemeris_grouped = ephemeris.groupby(by=["observatory_code", "mjd_utc"])
            ephemeris_split = [ephemeris_grouped.get_group(g).copy() for g in ephemeris_grouped.groups]

            test_orbit_ephemeris_grouped = test_orbit_ephemeris.groupby(by=["observatory_code", "mjd_utc"])
            test_orbit_ephemeris_split = [test_orbit_ephemeris_grouped.get_group(g) for g in test_orbit_ephemeris_grouped.groups]

            if threads > 1:

                if USE_RAY:
                
                    p = []
                    for e, te in zip(ephemeris_split, test_orbit_ephemeris_split):
                        p.append(projectephemeris_worker.remote(e, te))
                    ephemeris_dfs = ray.get(p)

                else:
                    p = mp.Pool(
                        processes=threads,
                        initializer=_init_worker,
                    ) 
                    
                    ephemeris_dfs = p.starmap(
                        projectephemeris_worker, 
                        zip(
                            ephemeris_split,
                            test_orbit_ephemeris_split
                        ) 
                    )
                    p.close()  
               
            else:
                ephemeris_dfs = []
                for e, te in zip(ephemeris_split, test_orbit_ephemeris_split):
                    ephemeris_df = projectephemeris_worker(e, te)
                    ephemeris_dfs.append(ephemeris_df)
            
            ephemeris = pd.concat(ephemeris_dfs)
            ephemeris.reset_index(
                drop=True,
                inplace=True
            )

        if shutdown:
            ray.shutdown()

        return ephemeris

    def _initialOrbitDetermination(self, observations, linkage_members, threads=NUM_THREADS, chunk_size=10, **kwargs):
        """
        Given observations and 

        THIS FUNCTION SHOULD BE DEFINED BY THE USER.
        
        """
        err = (
            "This backend does not have initial orbit propagation implemented."
        )
        raise NotImplementedError(err)

    def _orbitDetermination(self):
        err = (
            "This backend does not have orbit determination implemented."
        )
        raise NotImplementedError(err)

    def orbitDetermination(self, observations, threads=NUM_THREADS, chunk_size=10, **kwargs):
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


        """
        unique_objs = observations["obj_id"].unique()
        observations_split = [observations[observations["obj_id"].isin(unique_objs[i:i+chunk_size])].copy() for i in range(0, len(unique_objs), chunk_size)]
        backend_duplicated = [copy.deepcopy(self) for i in range(len(observations_split))]

        if USE_RAY:
            shutdown = False
            if not ray.is_initialized():
                ray.init(num_cpus=threads)
                shutdown = True
        
            od = []
            for o,  b in zip(observations_split, backend_duplicated):
                od.append(orbitdetermination_worker.remote(o, b))
            od_orbits_dfs = ray.get(od)

            if shutdown:
                ray.shutdown()
        else:
            p = mp.Pool(
                processes=threads,
                initializer=_init_worker,
            ) 
            
            od_orbits_dfs = p.starmap(
                orbitdetermination_worker, 
                zip(
                    observations_split,
                    backend_duplicated,
                ) 
            )
            p.close()  

        od_orbits = pd.concat(od_orbits_dfs)
        od_orbits.reset_index(
            drop=True,
            inplace=True
        )
        return od_orbits, residuals

    def _getObserverState(self, observers, origin="heliocenter"):
        err = (
            "This backend does not have observer state calculations implemented."
        )
        raise NotImplementedError(err)