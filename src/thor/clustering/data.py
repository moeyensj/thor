import hashlib
import logging
import uuid
from typing import List, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates import Origin
from adam_core.coordinates.residuals import Residuals
from adam_core.time import Timestamp

from ..projections import GnomonicCoordinates

logger = logging.getLogger(__name__)


def hash_obs_ids(obs_ids: List[str]) -> str:
    """
    Create unique strings for each set unique set of observation IDs

    We use hashes rather than original string in order to save memory.
    """
    return hashlib.md5("".join(sorted(set(obs_ids))).encode()).hexdigest()


def drop_duplicate_clusters(
    clusters: Union["Clusters", "FittedClusters"],
    cluster_members: Union["ClusterMembers", "FittedClusterMembers"],
) -> Tuple[Union["Clusters", "FittedClusters"], Union["ClusterMembers", "FittedClusterMembers"]]:
    """
    Drop clusters that have identical sets of observation IDs.

    Parameters
    ----------
    clusters: `~thor.clustering.data.Clusters`
        A table of clusters. Must be sorted by cluster_id.
    cluster_members: `~thor.clustering.data.ClusterMembers`
        A table of cluster members. Must be sorted by cluster_id.

    Returns
    -------
    `~thor.clustering.data.Clusters`, `~thor.clustering.data.ClusterMembers`
        A table of clusters with duplicate clusters removed.
        The cluster members belonging to those clusters.
    """
    if isinstance(clusters, ray.ObjectRef):
        clusters = ray.get(clusters)
    if isinstance(cluster_members, ray.ObjectRef):
        cluster_members = ray.get(cluster_members)

    if len(clusters) == 0 or len(cluster_members) == 0:
        return type(clusters).empty(), type(cluster_members).empty()

    # Ensure clusters and cluster members are sorted by cluster id
    # by spot checking the first few and last few rows are
    # in sorted order
    assert clusters.cluster_id[:3].to_pylist() == sorted(
        clusters.cluster_id[:3].to_pylist()
    ), "clusters must be sorted by cluster_id"  # noqa: E501
    assert clusters.cluster_id[-3:].to_pylist() == sorted(
        clusters.cluster_id[-3:].to_pylist()
    ), "clusters must be sorted by cluster_id"  # noqa: E501
    assert cluster_members.cluster_id[:3].to_pylist() == sorted(
        cluster_members.cluster_id[:3].to_pylist()
    ), "cluster_members must be sorted by cluster_id"  # noqa: E501
    assert cluster_members.cluster_id[-3:].to_pylist() == sorted(
        cluster_members.cluster_id[-3:].to_pylist()
    ), "cluster_members must be sorted by cluster_id"  # noqa: E501

    # We used to use a group by in pyarrow here,
    # but found the memory accumulationw as too high.
    # A simple loop that accumulates the distinct obs ids
    # for each cluster is more memory efficient.
    logger.info("Accumulating cluster observation IDs into single strings.")
    obs_ids_per_cluster: Union[List[str], pa.Array] = []
    current_obs_ids: List[str] = []
    current_cluster_id = None
    for member in cluster_members:
        cluster_id = member.cluster_id.to_pylist()[0]
        obs_id = member.obs_id.to_pylist()[0]
        if cluster_id != current_cluster_id:
            if current_cluster_id is not None:
                obs_ids_per_cluster.append(hash_obs_ids(current_obs_ids))
            current_cluster_id = cluster_id
            current_obs_ids = []
        current_obs_ids.append(obs_id)
    obs_ids_per_cluster.append(hash_obs_ids(current_obs_ids))

    logger.info("Grouping by unique observation sets.")
    obs_ids_per_cluster = pa.table(
        {
            "index": pa.array(np.arange(0, len(obs_ids_per_cluster))),
            "obs_ids": obs_ids_per_cluster,
        }
    )

    obs_ids_per_cluster = obs_ids_per_cluster.combine_chunks()
    obs_ids_per_cluster = obs_ids_per_cluster.group_by(["obs_ids"], use_threads=False)

    logger.info("Taking first index of each unique observation set.")
    indices = obs_ids_per_cluster.aggregate([("index", "first")])["index_first"]
    del obs_ids_per_cluster
    indices = indices.combine_chunks()

    logger.info("Taking clusters that belong to unique observation sets.")
    clusters = clusters.take(indices)

    logger.info("Taking cluster members that belong to unique clusters.")
    cluster_members = cluster_members.apply_mask(pc.is_in(cluster_members.cluster_id, clusters.cluster_id))
    return clusters, cluster_members


class Clusters(qv.Table):
    cluster_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    vtheta_x = qv.Float64Column()
    vtheta_y = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()


class ClusterMembers(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()


class FittedClusters(qv.Table):
    cluster_id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    test_orbit_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    theta_x0 = qv.Float64Column()
    theta_y0 = qv.Float64Column()
    vtheta_x = qv.Float64Column()
    vtheta_y = qv.Float64Column()
    atheta_x = qv.Float64Column()
    atheta_y = qv.Float64Column()
    origin = Origin.as_column()
    frame = qv.StringAttribute(default="testorbit")
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    rchi2 = qv.Float64Column()

    def evaluate(self, times: Timestamp) -> Tuple[pa.Array, GnomonicCoordinates]:
        """
        Propagate cluster positions to given times using the fitted polynomial model.

        The polynomial motion model is:
            theta_x(t) = 0.5 * atheta_x * dt² + vtheta_x * dt + theta_x0
            theta_y(t) = 0.5 * atheta_y * dt² + vtheta_y * dt + theta_y0

        where dt = t - t0 (in days).

        Parameters
        ----------
        times : Timestamp
            Times to propagate to.

        Returns
        -------
        cluster_ids : pa.Array
            Cluster IDs for each time.
        coords : GnomonicCoordinates
            Gnomonic coordinates for each time.

        Examples
        --------
        >>> # Evaluate clusters to given times
        >>> cluster_ids, coords = clusters.evaluate(times)
        """
        times_stacked = qv.concatenate([times for i in range(len(self))])
        origin_stacked = pa.concat_arrays(
            [pa.repeat(self.origin.code[i], len(times)) for i in range(len(self))]
        )
        epochs_mjd_stacked = np.repeat(
            self.time.rescale("tdb").mjd().to_numpy(zero_copy_only=False), len(times)
        )
        x0_stacked = np.repeat(self.theta_x0.to_numpy(zero_copy_only=False), len(times))
        y0_stacked = np.repeat(self.theta_y0.to_numpy(zero_copy_only=False), len(times))
        ax_stacked = np.repeat(self.atheta_x.to_numpy(zero_copy_only=False), len(times))
        ay_stacked = np.repeat(self.atheta_y.to_numpy(zero_copy_only=False), len(times))
        vx_stacked = np.repeat(self.vtheta_x.to_numpy(zero_copy_only=False), len(times))
        vy_stacked = np.repeat(self.vtheta_y.to_numpy(zero_copy_only=False), len(times))

        dt = times_stacked.rescale("tdb").mjd().to_numpy(zero_copy_only=False) - epochs_mjd_stacked
        x = 0.5 * ax_stacked * dt**2 + vx_stacked * dt + x0_stacked
        y = 0.5 * ay_stacked * dt**2 + vy_stacked * dt + y0_stacked
        cluster_ids = np.repeat(self.cluster_id.to_numpy(zero_copy_only=False), len(times))

        coords = GnomonicCoordinates.from_kwargs(
            theta_x=x,
            theta_y=y,
            time=times_stacked,
            origin=Origin.from_kwargs(code=origin_stacked),
            frame="testorbit",
        )
        return cluster_ids, coords


class FittedClusterMembers(qv.Table):
    cluster_id = qv.LargeStringColumn()
    obs_id = qv.LargeStringColumn()
    test_orbit_id = qv.LargeStringColumn(nullable=True)
    residuals = Residuals.as_column()
