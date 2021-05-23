import pytest
import numpy as np

from ..clusters import (
    _find_runs,
    _adjust_labels,
    _build_label_aliases,
    _sort_order_2d,
    _extend_2d_array,
    _label_clusters
)


def test_find_runs_nearmiss():
    points = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [2],
            [0],
        ],
        dtype=np.float64,
    )
    runs = _find_runs(points, min_samples=4)

    assert (runs == expected).all()

def test_find_runs_all_in_runs():
    points = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1, 2],
            [0, 0],
        ],
        dtype=np.float64,
    )
    runs = _find_runs(points, min_samples=4)
    assert (runs == expected).all()

def test_find_runs_longer_than_min_samples():
    points = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1, 2],
            [0, 0],
        ],
        dtype=np.float64,
    )
    runs = _find_runs(points, min_samples=2)
    assert (runs == expected).all()


def test_find_runs_changing_y():
    points = np.array(
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [2],
            [1],
        ],
        dtype=np.float64,
    )
    runs = _find_runs(points, min_samples=4)
    assert (runs == expected).all()


def test_find_runs_manyruns():
    """ Test case with enough runs to trigger array expansion """
    points = np.array(
        [
            [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            ],
            [
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            ],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3,
            ],
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8,
                0, 1, 2, 3, 4, 5, 6, 7, 8,
            ],
        ],
        dtype=np.float64,
    )
    runs = _find_runs(points, min_samples=2, expected_n_clusters=4)
    assert (runs == expected).all()


def test_adjust_labels():
    labels = np.array([-1, 0, -1, -1, 2, 2, 3, 3, -1])
    _adjust_labels(labels, 5)
    expected = np.array([-1, 5, -1, -1, 7, 7, 8, 8, -1])
    np.testing.assert_array_equal(expected, labels)


def test_build_label_aliases():
    labels1 = np.array([-1,  0,  1, -1,  1, -1, -1, -1])
    labels2 = np.array([ 2, -1,  2,  3,  4,  3, -1, -1])
    labels3 = np.array([ 5, -1,  5,  6,  7, -1,  8, -1])
    labels4 = np.array([-1, -1,  9, 10, 11, 12, 13, 14])

    n = 7

    aliases = _build_label_aliases(labels1, labels2, labels3, labels4, n)

    expected = {
        2: 1,
        4: 1,
        5: 1,
        6: 3,
        7: 1,
        9: 1,
        10: 3,
        11: 1,
        12: 3,
        13: 8,
    }
    assert expected == dict(aliases)


def test_sort_order_2d():
    points = np.array([
              # idx: sorted position
        [0, 1],  # 0: 1
        [1, 0],  # 1: 2
        [3, 0],  # 2: 5
        [0, 0],  # 3: 0
        [1, 2],  # 4: 3
        [2, 3],  # 5: 4
    ])
    points = points.T

    so = _sort_order_2d(points)
    expected = np.array([3, 0, 1, 4, 5, 2])
    np.testing.assert_array_equal(expected, so)


def test_extend_2d_array():
    points = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ])
    extended = _extend_2d_array(points, 10)
    assert extended.shape == (2, 10)
    np.testing.assert_array_equal(points, extended[:, :5])


def test_label_clusters():
    points = np.array([
        [0, 1, 2, 3, 4, 1],
        [3, 4, 5, 6, 7, 4],
    ])
    hits = np.array([
        [1, 3],
        [4, 6]
    ])
    expected = np.array(
        [-1, 0, -1, 1, -1, 0]
    )
    labels = _label_clusters(hits, points)
    np.testing.assert_array_equal(expected, labels)
