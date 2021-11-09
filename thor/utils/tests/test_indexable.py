import pytest
import numpy as np

from ..indexable import Indexable

class TestIndexable(Indexable):

    def __init__(self, values):

        self.values = values

    def __len__(self):
        return len(self.values)

SLICES = [
    slice(0, 1, 1),   #  0
    slice(1, 2, 1),   #  1
    slice(8, 9, 1),   # -1
    slice(7, 8, 1),   # -2
    slice(0, 10, 1),
    slice(0, 5, 1),
    slice(5, 10, 1),
    slice(5, 10, -1),
    slice(0, 10, -1),
]

def test_Indexable_slicing_array():

    array = np.arange(0, 10)

    indexable = TestIndexable(array)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values, indexable.values[s])

    return

def test_Indexable_slicing_marray():

    masked_array = np.ma.arange(0, 10)
    masked_array.mask = np.zeros(len(masked_array))
    masked_array.mask[0:10:2] = 1

    indexable = TestIndexable(masked_array)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values.data, indexable.values[s].data)
        np.testing.assert_equal(indexable[s].values.mask, indexable.values[s].mask)

    return

def test_Indexable_slicing_list():

    list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    indexable = TestIndexable(list)
    for s in SLICES:
        np.testing.assert_equal(indexable[s].values, indexable.values[s])

    return

def test_Indexable_iteration_array():

    array = np.arange(0, 10)

    indexable = TestIndexable(array)
    for i, ind in enumerate(indexable):
        np.testing.assert_equal(ind.values[0], array[i])

    return

def test_Indexable_iteration_marray():

    masked_array = np.ma.arange(0, 10)
    masked_array.mask = np.zeros(len(masked_array))
    masked_array.mask[0:10:2] = 1

    indexable = TestIndexable(masked_array)
    for i, ind in enumerate(indexable):
        np.testing.assert_equal(ind.values.data[0], masked_array.data[i])
        np.testing.assert_equal(ind.values.mask[0], masked_array.mask[i])

    return

def test_Indexable_iteration_list():

    list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    indexable = TestIndexable(list)
    for i, ind in enumerate(indexable):
        np.testing.assert_equal(ind.values[0], list[i])

    return

def test_Indexable_deletion_array():

    array = np.arange(0, 10)

    indexable = TestIndexable(array)
    del indexable[5]
    np.testing.assert_equal(indexable.values, np.array([0, 1, 2, 3, 4, 6, 7, 8, 9]))

    del indexable[-1]
    np.testing.assert_equal(indexable.values, np.array([0, 1, 2, 3, 4, 6, 7, 8]))

    del indexable[1:4]
    np.testing.assert_equal(indexable.values, np.array([0, 4, 6, 7, 8]))
    return

def test_Indexable_deletion_marray():

    masked_array = np.ma.arange(0, 10)
    masked_array.mask = np.zeros(len(masked_array))
    masked_array.mask[0:10:2] = 1

    indexable = TestIndexable(masked_array)
    del indexable[5]
    np.testing.assert_equal(indexable.values.data, np.array([0, 1, 2, 3, 4, 6, 7, 8, 9]))
    np.testing.assert_equal(indexable.values.mask, np.array([True, False, True, False, True, True, False, True, False]))

    del indexable[-1]
    np.testing.assert_equal(indexable.values.data, np.array([0, 1, 2, 3, 4, 6, 7, 8]))
    np.testing.assert_equal(indexable.values.mask, np.array([True, False, True, False, True, True, False, True]))

    del indexable[1:4]
    np.testing.assert_equal(indexable.values.data, np.array([0, 4, 6, 7, 8]))
    np.testing.assert_equal(indexable.values.mask, np.array([True, True, True, False, True]))

    return

def test_Indexable_raises():
    # Tuples are not supported by the Indexable class
    with pytest.raises(NotImplementedError):
        indexable = TestIndexable((0, 1, 2, 3, 4, 5))
        indexable[2]

    return
