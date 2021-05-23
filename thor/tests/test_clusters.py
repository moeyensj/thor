import pytest
import numpy as np

from ..clusters import _find_runs, _find_clusters_hotspots_2d


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
