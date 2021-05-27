import pandas as pd

from ..config import Config
from ..utils import _checkTime
from ..backend import PYOORB
from ..backend import FINDORB
from ..backend import MJOLNIR

__all__ = [
    "propagateOrbits"
]

def propagateOrbits(
        orbits,
        t1,
        backend="MJOLNIR",
        backend_kwargs={},
        num_jobs=Config.NUM_JOBS,
        chunk_size=1
    ):
    """
    Propagate orbits using desired backend.

    To insure consistency, propagated epochs are always returned in TDB regardless of the backend.

    Parameters
    ----------
    orbits : `~numpy.ndarray` (N, 6)
        Orbits to propagate. If backend is 'THOR', then these orbits must be expressed
        as heliocentric ecliptic cartesian elements. If backend is 'PYOORB' orbits may be
        expressed in heliocentric keplerian, cometary or cartesian elements.
    t1 : `~astropy.time.core.Time` (M)
        Epochs to which to propagate each orbit.
    backend : {'MJOLNIR', 'PYOORB', 'FINDORB'}, optional
        Which backend to use.
    backend_kwargs : dict, optional
        Settings and additional parameters to pass to selected
        backend.

    Returns
    -------
    propagated_orbits : `~pandas.DataFrame` (N x M, 8)
        A DataFrame containing the heliocentric propagated orbits.
    """
    # Check that t1 is an astropy.time objects
    _checkTime(t1, "t1")

    if backend == "MJOLNIR":
        backend = MJOLNIR(**backend_kwargs)

    elif backend == "PYOORB":
        backend = PYOORB(**backend_kwargs)

    elif backend == "FINDORB":
        backend = FINDORB(**backend_kwargs)

    else:
        err = (
            "backend should be one of 'MJOLNIR', 'PYOORB', 'FINDORB'"
        )
        raise ValueError(err)

    propagated = backend.propagateOrbits(
        orbits,
        t1,
        num_jobs=num_jobs,
        chunk_size=chunk_size
    )
    return propagated