import pytest
import numpy as np

from ..coordinates import _ingest_coordinate


def test__ingest_coordinate():
    # Create 6 random arrays and test that
    # _ingest_coordinate correctly places these
    # arrays into the returned masked array
    N, D = 1000, 6

    coord_arrays = []
    for d in range(D):
        coord_arrays.append(np.random.rand(N))

    coords = None
    for d, q in enumerate(coord_arrays):
        coords = _ingest_coordinate(q, d, coords=coords)

    for d in range(D):
        np.testing.assert_equal(coords[:, d], coord_arrays[d])

    return


def test__ingest_coordinate_raises():
    # Create 2 random arrays of varying lengths
    # and test that _ingest_coordinate raises
    # a ValueError
    N1 = 500
    N2 = 501

    coord_arrays = []
    for d, n in enumerate([N1, N2]):
        coord_arrays.append(np.random.rand(n))

    coords = None
    for d, q in enumerate(coord_arrays):
        if d == 0:
            coords = _ingest_coordinate(q, d, coords=coords)
        else:
            with pytest.raises(ValueError):
                coords = _ingest_coordinate(q, d, coords=coords)

    return


def test__ingest_coordinate_masks():
    # Create 6 random arrays with varying NaN values
    # to represent missing measurements and test that
    # _ingest_coordinate correctly places these
    # arrays into the returned masked array
    N, D = 1000, 6

    coord_arrays = []
    mask_arrays = []
    for d in range(D):
        q = np.random.rand(N)
        inds = np.random.choice(np.arange(0, N), 50, replace=False)
        q[inds] = np.NaN
        mask = np.zeros(N, dtype=bool)
        mask[inds] = 1

        coord_arrays.append(q)
        mask_arrays.append(mask)

    coords = None
    for d, q in enumerate(coord_arrays):
        coords = _ingest_coordinate(q, d, coords=coords)

    for d in range(D):
        np.testing.assert_equal(coords[:, d], coord_arrays[d])
        np.testing.assert_equal(coords.mask[:, d], mask_arrays[d])

    return
