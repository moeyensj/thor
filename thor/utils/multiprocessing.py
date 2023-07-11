import logging
import multiprocessing as mp
import signal

import numpy as np
import pandas as pd

__all__ = ["Timeout", "yieldChunks", "calcChunkSize", "_initWorker", "_checkParallel"]

logger = logging.getLogger(__name__)


class Timeout:
    ### Taken from https://stackoverflow.com/a/22348885
    def __init__(self, seconds=30, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def yieldChunks(indexable, chunk_size):
    """
    Generator that yields chunks of size chunk_size.

    Parameters
    ----------
    indexable : list, `~numpy.ndarray`, `~pandas.DataFrame`, `~pandas.Series` (N)
        Indexable object that needs to be divided into
        chunks.
    chunk_size : int
        Size of each chunk.

    Yields
    ------
    chunk : indexable (<=chunk_size)
        Chunks of indexable
    """
    if isinstance(indexable, list) or isinstance(indexable, np.ndarray):
        for c in range(0, len(indexable), chunk_size):
            yield indexable[c : c + chunk_size]
    elif isinstance(indexable, pd.DataFrame) or isinstance(indexable, pd.Series):
        for c in range(0, len(indexable), chunk_size):
            yield indexable.iloc[c : c + chunk_size]
    else:
        err = "Indexable should be one of {list, `~numpy.ndarray`, `~pandas.DataFrame`, `~pandas.Series`}"
        raise ValueError(err)


def calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=1):
    """
    Calculate the optimal chunk size such that each worker gets at
    least min_chunk_size chunks but does not get more than max_chunk_size
    chunks. The goal is for no worker to be idle in terms of the number of items
    it recieves

    Parameters
    ----------
    n : int
        Number of items.
    num_workers : int
        Number of workers to which items will be distributed.
    max_chunk_size : int
        Maximum chunk size to be given to each worker.
    min_chunk_size : int, optional
        Minimum chunk size to be given to each worker.

    Yields
    ------
    chunk_size : int
        Chunk_size between min_chunk_size and max_chunk_size.
    """
    # Calculate the number of n that should be sent to each worker
    c = np.maximum(np.floor(n / num_workers), min_chunk_size).astype(int)

    # Make sure this number does not exceed the maximum chunk size
    chunk_size = np.minimum(c, max_chunk_size)
    return chunk_size


def _initWorker():
    """
    Tell multiprocessing worker to ignore signals, will only
    listen to parent process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return


def _checkParallel(num_jobs, parallel_backend):
    """
    Helper function to determine how many workers (jobs) should be submitted.
    If the parallelization backend is Python's multiprocessing ('mp') and num_jobs
    is either "auto" or None, then mp.cpu_count() will be used to determine the number
    of jobs. If num_jobs is not "auto" or None, then that number will be used instead.
    If the parallelization backend is ray, then the number of resources on the cluster will
    determine the number of workers, and num_jobs is ignored.

    Parameters
    ----------
    num_jobs : {None, "auto", int}
        Number of jobs to launch.
    parallel_backend : str
        Name of backend. Should be one of {'ray', 'mp', 'cf'}.

    Returns
    -------
    enable_parallel : bool
        True if parallelization should be used (true for cases where num_jobs > 1).
    num_workers : int
        The number of workers to use.

    Raises
    ------
    ValueError : If parallel_backend is not one of {'ray', 'mp', 'cf'}.
    """
    if isinstance(num_jobs, str) or (num_jobs is None) or (num_jobs > 1):

        # Check that pareallel_backend is one of the support types
        backends = ["ray", "mp", "cf"]
        if parallel_backend not in backends:
            err = "parallel_backend should be one of {'ray', 'mp', 'cf'}"
            raise ValueError(err)

        enable_parallel = True
        if parallel_backend == "ray":
            import ray

            if not num_jobs == "auto" or num_jobs is not None:
                logger.warning(
                    "This process is running with the ray parallelization backend: num_jobs parameter will be ignored."
                )

            num_workers = int(ray.cluster_resources()["CPU"])

        else:
            if num_jobs == "auto" or num_jobs is None:
                num_workers = mp.cpu_count()
            else:
                num_workers = num_jobs
    else:
        num_workers = 1
        enable_parallel = False

    return enable_parallel, num_workers
