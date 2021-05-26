import numpy as np

__all__ = [
    "yieldChunks",
    "calcChunkSize"
]

def yieldChunks(l, chunk_size):
    """
    Generator that yields chunks of size chunk_size.

    Parameters
    ----------
    l : list (N)
        List or list-like object that needs to be divided into
        chunks
    chunk_size : int
        Size of each chunk.

    Yields
    ------
    chunk : list (<=chunk_size)
        Chunk of input list
    """
    for c in range(0, len(l), chunk_size):
        yield l[c : c + chunk_size]

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
