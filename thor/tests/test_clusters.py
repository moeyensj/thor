import pytest
import numpy as np

from ..clusters import _find_runs, _find_clusters_hotspots_2d


def test_find_clusters_hotspot2d():
    eps = 0.1
    # Cluster 1: 5 points that are centered at -5.55, +6.25
    cluster1_lo = [-5.6, 6.2]
    cluster1_hi = [-5.5, 6.3]
    cluster1 = np.random.uniform(cluster1_lo, cluster1_hi, (5, 2))

    # Cluster 2: 3 points that are centered at 0.05, 0.05
    cluster2_lo = [0.0, 0.0]
    cluster2_hi = [0.1, 0.1]
    cluster2 = np.random.uniform(cluster2_lo, cluster2_hi, (3, 2))

    # Cluster 3: 6 points that are centered at 8.05, 6.25
    cluster3_lo = [8.0, 6.2]
    cluster3_hi = [8.1, 6.3]
    cluster3 = np.random.uniform(cluster3_lo, cluster3_hi, (6, 2))

    # Finally, another 100 random points in [-10, 10]
    xy = np.random.uniform([-10, -10], [10, 10], (100, 2))

    # Remove any points that are accidentally in our clusters, just for test
    # consistency.
    for lo, hi in [(cluster1_lo, cluster1_hi), (cluster2_lo, cluster2_hi), (cluster3_lo, cluster3_hi)]:
        mask = ((xy > lo) & (xy < hi)).all(1)
        xy = xy[~mask]

    data = np.append(cluster1, cluster2, axis=0)
    data = np.append(data, cluster3, axis=0)
    data = np.append(data, xy, axis=0)
    np.random.shuffle(data.T)

    def cluster_present(cluster, cluster_list):
        """ little helper to see if cluster is present in cluster_list"""
        found = False
        for c in cluster_list:
            if c.shape[0] == cluster.shape[0]:
                if (np.sort(data[c]) == np.sort(cluster)).all():
                    found = True
        return found

    # Search for clusters of size at least 5. We should find cluster 1 and 3.
    clusters = _find_clusters_hotspots_2d(data, eps, 5)
    assert len(clusters) == 2
    assert cluster_present(cluster1, clusters)
    assert cluster_present(cluster3, clusters)

    # Only cluster 3 has size 6.
    clusters = _find_clusters_hotspots_2d(data, eps, 6)
    assert len(clusters) == 1
    assert cluster_present(cluster3, clusters)

    # All three clusters can be found with size 3.
    clusters = _find_clusters_hotspots_2d(data, eps, 3)
    assert len(clusters) == 3
    assert cluster_present(cluster1, clusters)
    assert cluster_present(cluster2, clusters)
    assert cluster_present(cluster3, clusters)


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
