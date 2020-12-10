import os
import copy
import signal
import pandas as pd
import multiprocessing as mp

from ..config import Config

USE_RAY = Config.USE_RAY
NUM_THREADS = Config.NUM_THREADS

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

__all__ = ["Backend"]

def init_worker():
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

if USE_RAY:
    import ray
    propagation_worker = ray.remote(propagation_worker)
    ephemeris_worker = ray.remote(ephemeris_worker)

class Backend:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
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
                initializer=init_worker,
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

    def generateEphemeris(self, orbits, observers, threads=NUM_THREADS, chunk_size=100):
        """
        Generate ephemerides for each orbit in orbits as observed by each observer
        in observers.

        Parameters
        ----------
        orbits : `~thor.orbits.orbits.Orbits`
            Orbits for which to generate ephemerides.
        observers : dict or `~pandas.DataFrame`
            A dictionary with observatory codes as keys and observation_times (`~astropy.time.core.Time`) as values. 
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

            if shutdown:
                ray.shutdown()
        else:
            p = mp.Pool(
                processes=threads,
                initializer=init_worker,
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
        return ephemeris
        
    def _orbitDeterminaton(self):
        err = (
            "This backend does not have orbit determination implemented."
        )
        raise NotImplementedError(err)

    def _getObserverState(self, observers, origin="heliocenter"):
        err = (
            "This backend does not have observer state calculations implemented."
        )
        raise NotImplementedError(err)