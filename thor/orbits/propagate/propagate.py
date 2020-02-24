import numpy as np
import pandas as pd

from ...constants import Constants as c
from ...utils import _checkTime
from .universal import propagateUniversal
from .pyoorb import propagateOrbitsPYOORB

__all__ = [
    "propagateOrbits"
]

MU = c.G * c.M_SUN

THOR_PROPAGATOR_KWARGS = {
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-15
}

PYOORB_PROPAGATOR_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "TT", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "N",
    "ephemeris_file" : "de430.dat"
}

def propagateOrbits(orbits, t0, t1, backend="THOR", backend_kwargs=None):
    """
    Propagate orbits using desired backend. 

    To insure consistency, propagated epochs are always returned in TDB regardless of the backend.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to propagate. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be 
        expressed in keplerian, cometary or cartesian elements.
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
        A DataFrame containing the propagated orbits.
    """
    # Check that both t0 and t1 are astropy.time objects
    _checkTime(t0, "t0")
    _checkTime(t1, "t1")

    # All propagations in THOR should be done with times in the TDB time scale
    t0_tdb = t0.tdb.mjd
    t1_tdb = t1.tdb.mjd

    if backend == "THOR":
        if backend_kwargs == None:
            backend_kwargs = THOR_PROPAGATOR_KWARGS

        propagated = propagateUniversal(orbits, t0_tdb, t1_tdb, **backend_kwargs)

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
        if backend_kwargs == None:
            backend_kwargs = PYOORB_PROPAGATOR_KWARGS

        # PYOORB does not support TDB, so set times to TT and add a TDB correction
        t0_tt = t0.tt.mjd + (t0.tdb.mjd - t0.tt.mjd)
        t1_tt = t1.tt.mjd + (t1.tdb.mjd - t1.tt.mjd)
        backend_kwargs["time_scale"] = "TT"
        
        propagated = propagateOrbitsPYOORB(orbits, t0_tt, t1_tt, **backend_kwargs) 

        # Convert return epoch back to TDB
        propagated.rename(columns={"epoch_mjd" : "epoch_mjd_tdb"}, inplace=True)
        epoch_mjd_tdb = [t1_tdb for i in range(len(orbits))]
        propagated["epoch_mjd_tdb"] = np.concatenate(epoch_mjd_tdb)

    else:
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)

    return propagated