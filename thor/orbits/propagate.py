import warnings

from ..utils import _check_times
from ..backend import Backend
from ..backend import PYOORB
from ..backend import FINDORB
from ..backend import MJOLNIR

__all__ = [
    "propagate_orbits"
]

def propagate_orbits(
        orbits,
        t1,
        backend="MJOLNIR",
        backend_kwargs={},
        chunk_size=1,
        num_jobs=1,
        parallel_backend="mp"
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
    chunk_size : int, optional
        Number of orbits to send to each job.
    num_jobs : int, optional
        Number of jobs to launch.
    parallel_backend : str, optional
        Which parallelization backend to use {'ray', 'mp'}. Defaults to using Python's multiprocessing
        module ('mp').

    Returns
    -------
    propagated_orbits : `~pandas.DataFrame` (N x M, 8)
        A DataFrame containing the heliocentric propagated orbits.
    """
    # Check that t1 is an astropy.time objects
    _check_times(t1, "t1")

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
            "backend should be one of 'MJOLNIR', 'PYOORB', 'FINDORB'"
        )
        raise ValueError(err)

    propagated = backend.propagate_orbits(
        orbits,
        t1,
        chunk_size=chunk_size,
        num_jobs=num_jobs,
        parallel_backend=parallel_backend
    )
    return propagated