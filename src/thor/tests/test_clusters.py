import numpy as np
import pytest

from ..clusters import (
    ClusterMembers,
    Clusters,
    _adjust_labels,
    _build_label_aliases,
    _extend_2d_array,
    _find_runs,
    _label_clusters,
    _sort_order_2d,
    drop_duplicate_clusters,
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
    """Test case with enough runs to trigger array expansion"""
    points = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
            ],
            [
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
                7,
                7,
                8,
                8,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
                7,
                7,
                8,
                8,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
                7,
                7,
                8,
                8,
            ],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
            ],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
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
    labels1 = np.array([-1, 0, 1, -1, 1, -1, -1, -1])
    labels2 = np.array([2, -1, 2, 3, 4, 3, -1, -1])
    labels3 = np.array([5, -1, 5, 6, 7, -1, 8, -1])
    labels4 = np.array([-1, -1, 9, 10, 11, 12, 13, 14])

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
    points = np.array(
        [
            # idx: sorted position
            [0, 1],  # 0: 1
            [1, 0],  # 1: 2
            [3, 0],  # 2: 5
            [0, 0],  # 3: 0
            [1, 2],  # 4: 3
            [2, 3],  # 5: 4
        ]
    )
    points = points.T

    so = _sort_order_2d(points)
    expected = np.array([3, 0, 1, 4, 5, 2])
    np.testing.assert_array_equal(expected, so)


def test_extend_2d_array():
    points = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ]
    )
    extended = _extend_2d_array(points, 10)
    assert extended.shape == (2, 10)
    np.testing.assert_array_equal(points, extended[:, :5])


def test_label_clusters():
    points = np.array(
        [
            [0, 1, 2, 3, 4, 1],
            [3, 4, 5, 6, 7, 4],
        ]
    )
    hits = np.array([[1, 3], [4, 6]])
    expected = np.array([-1, 0, -1, 1, -1, 0])
    labels = _label_clusters(hits, points)
    np.testing.assert_array_equal(expected, labels)


def test_Clusters_drop_duplicates():
    # Test that the cluster deduplication works as expected
    # Here we duplicate the same 5 clusters 10000 times and check that the
    # deduplication correctly identifies the first 5 clusters
    obs_ids = [
        ["obs_01", "obs_02", "obs_03", "obs_04", "obs_05"],
        ["obs_02", "obs_03", "obs_04", "obs_05", "obs_06"],
        ["obs_03", "obs_04", "obs_05", "obs_06", "obs_07"],
        ["obs_04", "obs_05", "obs_06", "obs_07", "obs_08"],
        ["obs_05", "obs_06", "obs_07", "obs_08", "obs_09"],
    ]

    obs_ids_duplicated = []
    for i in range(10000):
        obs_ids_duplicated += obs_ids
    cluster_ids = [f"c{i:05d}" for i in range(len(obs_ids_duplicated))]

    clusters = Clusters.from_kwargs(
        cluster_id=cluster_ids,
        vtheta_x=np.full(len(cluster_ids), 0.0),
        vtheta_y=np.full(len(cluster_ids), 0.0),
        arc_length=np.full(len(cluster_ids), 0.0),
        num_obs=np.full(len(cluster_ids), 5),
    )
    cluster_members = ClusterMembers.from_kwargs(
        cluster_id=list(np.repeat(cluster_ids, 5)),
        obs_id=[obs for cluster_members_i in obs_ids_duplicated for obs in cluster_members_i],
    )

    clusters_filtered, cluster_members_filtered = drop_duplicate_clusters(clusters, cluster_members)
    assert len(clusters_filtered) == 5
    assert clusters_filtered.cluster_id.to_pylist() == [
        "c00000",
        "c00001",
        "c00002",
        "c00003",
        "c00004",
    ]

    assert len(cluster_members_filtered) == 25
    np.testing.assert_equal(
        cluster_members_filtered.cluster_id.to_numpy(zero_copy_only=False),
        np.repeat(cluster_ids[:5], 5),
    )
    np.testing.assert_equal(
        cluster_members_filtered.obs_id.to_numpy(zero_copy_only=False),
        np.hstack(np.array(obs_ids)),
    )


def test_drop_duplicate_clusters_sorted():
    """
    Test that drop duplicate clusters throws an assertion error if not sorted
    """
    clusters = Clusters.from_kwargs(
        cluster_id=["c00005", "c00000", "c00001", "c00002", "c00003", "c00004"],
        vtheta_x=np.full(6, 0.0),
        vtheta_y=np.full(6, 0.0),
        arc_length=np.full(6, 0.0),
        num_obs=np.full(6, 5),
    )

    cluster_members = ClusterMembers.from_kwargs(
        cluster_id=["c00005", "c00000", "c00001", "c00002", "c00003", "c00004"],
        obs_id=[
            "obs_01",
            "obs_02",
            "obs_03",
            "obs_04",
            "obs_05",
            "obs_06",
        ],
    )

    with pytest.raises(AssertionError):
        drop_duplicate_clusters(clusters, cluster_members)

    clusters = clusters.sort_by([("cluster_id", "ascending")])
    cluster_members = cluster_members.sort_by([("cluster_id", "ascending")])

    drop_duplicate_clusters(clusters, cluster_members)
