import numpy as np
import pandas as pd
from astropy.time import Time

from ..utils import _checkTime
from ..backend import PYOORB
from .state import shiftOrbitsOrigin
from .handler import _backendHandler
from .universal_propagate import propagateUniversal

__all__ = [
    "propagateOrbits"
]

def propagateOrbits(orbits, t0, t1, backend="THOR", backend_kwargs=None):
    """
    Propagate orbits using desired backend. 

    To insure consistency, propagated epochs are always returned in TDB regardless of the backend.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to propagate. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be 
        expressed in heliocentric keplerian, cometary or cartesian elements.
    t0 : `~astropy.time.core.Time` (N)
        Epoch at which orbits are defined.
    t1 : `~astropy.time.core.Time` (M)
        Epochs to which to propagate each orbit.
    backend : {'THOR', 'PYOORB'}, optional
        Which backend to use. 
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected 
        backend.

    Returns
    -------
    propagated_orbits : `~pandas.DataFrame` (N x M, 8)
        A DataFrame containing the heliocentric propagated orbits.
    """
    # Check that both t0 and t1 are astropy.time objects
    _checkTime(t0, "t0")
    _checkTime(t1, "t1")

    # All propagations in THOR should be done with times in the TDB time scale
    t0_tdb = t0.tdb.mjd
    t1_tdb = t1.tdb.mjd

    if backend_kwargs is None:
        backend_kwargs = _backendHandler(backend, "propagate")

    if backend == "THOR":
        origin = backend_kwargs.pop("origin")

        if origin == "barycenter":
            # Shift orbits to barycenter
            orbits_ = shiftOrbitsOrigin(orbits, t0, 
                origin_in="heliocenter",
                origin_out="barycenter")

        elif origin == "heliocenter":
            orbits_ = orbits
            
        else:
            err = (
                "origin should be one of {'heliocenter', 'barycenter'}"
            )
            raise ValueError(err)

        propagated = propagateUniversal(orbits_, t0_tdb, t1_tdb, **backend_kwargs)

        if origin == "barycenter":
            t1_tdb_stacked = Time(propagated[:, 1], scale="tdb", format="mjd")
            propagated[:, 2:] = shiftOrbitsOrigin(propagated[:, 2:], t1_tdb_stacked, 
                origin_in="barycenter",
                origin_out="heliocenter")

        backend_kwargs["origin"] = origin

        propagated = pd.DataFrame(
            propagated,
            columns=[
                "orbit_id",
                "epoch_mjd_tdb",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
            ]
        )
        propagated["orbit_id"] = propagated["orbit_id"].astype(int)

    elif backend == "PYOORB":
        
        PYOORB_CONFIG = {
            "orbit_type" : "cartesian", 
            "magnitude" : 20, 
            "slope" : 0.15, 
            "dynamical_model" : "N",
            "ephemeris_file" : "de430.dat"
        }
        backend = PYOORB(**PYOORB_CONFIG)
        propagated = backend.propagateOrbits(
            orbits,
            t0,
            t1
        )

    else:
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)

    return propagated