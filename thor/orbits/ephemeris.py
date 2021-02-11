import warnings
import pandas as pd

from ..config import Config
from ..backend import Backend
from ..backend import PYOORB
from ..backend import FINDORB
from ..backend import MJOLNIR

__all__ = [
    "generateEphemeris"
]

def generateEphemeris(
        orbits, 
        observers, 
        backend="MJOLNIR",
        backend_kwargs={},
        test_orbit=None,
        threads=Config.NUM_THREADS, 
        chunk_size=1
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
        (See: `~thor.observatories.getObserverState`)
    backend : {'MJOLNIR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    ephemeris : `~pandas.DataFrame` (N x M, 21) or (N x M, 18)
        A DataFrame containing the generated ephemeris.
    """
    if backend == "MJOLNIR":
        backend = MJOLNIR(**backend_kwargs)

    elif backend == "PYOORB":
        backend = PYOORB(**backend_kwargs)
        
    elif backend == "FINDORB":
        backend = FINDORB(**backend_kwargs)

    elif isinstance(backend, Backend):
        backend = backend

        if len(backend_kwargs) > 0:
            warnings.warn("backend_kwargs will be ignored since a instantiated backend class has been given.")

    else: 
        err = (
            "backend should be one of 'MJOLNIR', 'PYOORB', 'FINDORB' or an instantiated Backend class"
        )
        raise ValueError(err)

    ephemeris = backend.generateEphemeris(
        orbits,
        observers,
        test_orbit=test_orbit,
        threads=threads,
        chunk_size=chunk_size
    )
    ephemeris.sort_values(
        by=["orbit_id", "observatory_code", "mjd_utc"],
        inplace=True
    )
    ephemeris.reset_index(
        inplace=True,
        drop=True
    )
    return ephemeris

