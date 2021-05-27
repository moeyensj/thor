import pytest
import numpy as np
import pandas as pd

from ..multiprocessing import yieldChunks
from ..multiprocessing import calcChunkSize

def test_yieldChunks_list():
    # Create list of data
    indexable = [i for i in range(15)]

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(np.array(chunk), desired)
    return

def test_yieldChunks_array():
    # Create list of data
    indexable = np.arange(0, 15, 1)

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(chunk, desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk, desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(chunk, desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(chunk, desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk, desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(chunk, desired)
    return

def test_yieldChunks_series():
    # Create series of data
    indexable = pd.Series(np.arange(0, 15, 1))

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)
    return

def test_yieldChunks_series_offsetIndex():
    # Create series of data
    indexable = pd.Series(np.arange(0, 15, 1))
    # Offset the index and make sure chunking is done independent of
    # the values of the index
    indexable.index = np.arange(0, 150, 10)

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(chunk.values, desired)
    return

def test_yieldChunks_dataframe_offsetIndex():
    # Create dataframe of data
    data = {
        "x" : np.arange(0, 15, 1)
    }
    indexable = pd.DataFrame(data)
    # Offset the index and make sure chunking is done independent of
    # the values of the index
    indexable.index = np.arange(0, 150, 10)

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)
    return

def test_yieldChunks_dataframe():
    # Create dataframe of data
    data = {
        "x" : np.arange(0, 15, 1)
    }
    indexable = pd.DataFrame(data)

    # Set chunk_size to 10
    chunk_size = 10
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 9
    chunk = next(generator)
    desired = np.arange(0, 10, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Second iteration should yield 10 through 15
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Set chunk_size to 5
    chunk_size = 5
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 4
    chunk = next(generator)
    desired = np.arange(0, 5, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Second iteration should yield 5 through 9
    chunk = next(generator)
    desired = np.arange(5, 10, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Third iteration should yield 10 through 14
    chunk = next(generator)
    desired = np.arange(10, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)

    # Set chunk_size to 20
    chunk_size = 20
    generator = yieldChunks(indexable, chunk_size)

    # First iteration should yield 0 through 14
    chunk = next(generator)
    desired = np.arange(0, 15, 1)
    np.testing.assert_array_equal(chunk["x"].values, desired)
    return

def test_yieldChunks_errors():

    # Make sure yieldChunks raises an error for unsupported types
    chunk_size = 1
    with pytest.raises(ValueError):
        generator = yieldChunks(set(), chunk_size)
        next(generator)

    return

def test_calcChunkSize():

    num_workers = 60
    n = 100
    max_chunk_size = 10
    min_chunk_size = 1

    # Number of workers is less than 2x the number of things to process, so
    # chunk size should fall to the minimum value
    chunk_size = calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=min_chunk_size)
    assert chunk_size == 1

    min_chunk_size = 5
    # Number of workers is less than 2x the number of things to process, so
    # chunk size should fall to the minimum value
    chunk_size = calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=min_chunk_size)
    assert chunk_size == 5

    num_workers = 10
    n = 1000
    max_chunk_size = 10
    min_chunk_size = 1

    # Number of things to process is 10x the number of workers, the max chunk size is 10 so
    # the chunk_size should be 10
    chunk_size = calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=min_chunk_size)
    assert chunk_size == 10

    num_workers = 10
    n = 1000
    max_chunk_size = 5
    min_chunk_size = 1

    # Number of things to process is 10x the number of workers, the max chunk size is 5 so
    # the chunk_size should be 5
    chunk_size = calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=min_chunk_size)
    assert chunk_size == 5

    num_workers = 100
    n = 10000
    max_chunk_size = 1000
    min_chunk_size = 1

    # Number of things to process is 10x the number of workers, the max chunk size is 1000 so
    # the chunk_size should be 10000/100 = 100
    chunk_size = calcChunkSize(n, num_workers, max_chunk_size, min_chunk_size=min_chunk_size)
    assert chunk_size == 100
    return