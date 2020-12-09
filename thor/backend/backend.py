import os
import copy
import signal
import pandas as pd
import multiprocessing as mp

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

    def propagateOrbits(self, orbits, t1, threads=4, chunk_size=100):
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
            p = mp.Pool(
                processes=threads,
                initializer=init_worker,
            ) 
            orbits_split = orbits.split(chunk_size)
            t1_duplicated = [copy.deepcopy(t1) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]

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

    def generateEphemeris(self, orbits, observers, threads=4, chunk_size=100):
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
        if threads > 1:
            p = mp.Pool(
                processes=threads,
                initializer=init_worker,
            ) 
            orbits_split = orbits.split(chunk_size)
            observers_duplicated = [copy.deepcopy(observers) for i in range(len(orbits_split))]
            backend_duplicated = [copy.deepcopy(self) for i in range(len(orbits_split))]

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