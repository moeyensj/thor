import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.coordinates.residuals import Residuals
from adam_core.ray_cluster import initialize_use_ray
from adam_core.utils.iter import _iterate_chunks

from ..projections import GnomonicCoordinates
from ..range_and_transform import TransformedDetections
from .data import ClusterMembers, Clusters, FittedClusterMembers, FittedClusters

logger = logging.getLogger(__name__)


def fit_cluster(
    cluster: Clusters, cluster_members: ClusterMembers, transformed_detections: TransformedDetections
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Fit a cluster with a 2nd order polynomial motion model in theta_x and theta_y.

    Parameters
    ----------
    cluster : `~thor.clustering.data.Clusters`
        Cluster.
    cluster_members : `~thor.clustering.data.ClusterMembers`
        Cluster members.
    transformed_detections : `~thor.range_and_transform.TransformedDetections`
        Transformed detections.

    Returns
    -------
    fitted_cluster : `~thor.clustering.data.FittedClusters`
        Fitted cluster.
    fitted_cluster_members : `~thor.clustering.data.FittedClusterMembers`
        Fitted cluster members.
    """
    try:
        cluster_detections = transformed_detections.apply_mask(
            pc.is_in(transformed_detections.id, cluster_members.obs_id)
        )
        cluster_detections = cluster_detections.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        gnomonic_coords = cluster_detections.coordinates
        theta_x = gnomonic_coords.theta_x.to_numpy(zero_copy_only=False)
        theta_y = gnomonic_coords.theta_y.to_numpy(zero_copy_only=False)
        time = gnomonic_coords.time.mjd().to_numpy(zero_copy_only=False)

        # Use relative time from the first observation to avoid numerical issues
        # and make x0, y0 represent position at the first observation
        t0 = time[0]
        dt = time - t0  # days since first observation

        # Fit a 2nd order polynomial to the data as a function of relative time
        # theta(dt) = theta0 + v*dt + 0.5*a*dt²
        coords = np.empty((len(dt), 2))
        coords[:, 0] = theta_x
        coords[:, 1] = theta_y
        coeffs = np.polyfit(dt, coords, 2)

        # coeffs[0] is the quadratic coefficient in theta(dt) = c2*dt² + c1*dt + c0
        # Store atheta as physical acceleration, i.e., atheta = 2 * c2
        ax = 2.0 * coeffs[0, 0]  # deg/day²
        ay = 2.0 * coeffs[0, 1]
        vx = coeffs[1, 0]  # deg/day
        vy = coeffs[1, 1]
        x0 = coeffs[2, 0]  # deg (position at first observation)
        y0 = coeffs[2, 1]

        x_pred = np.polyval(coeffs[:, 0], dt)
        y_pred = np.polyval(coeffs[:, 1], dt)

        gnomonic_pred = GnomonicCoordinates.from_kwargs(
            time=gnomonic_coords.time,
            theta_x=x_pred,
            theta_y=y_pred,
            origin=gnomonic_coords.origin,
            frame=gnomonic_coords.frame,
        )

        residuals = Residuals.calculate(gnomonic_coords, gnomonic_pred, custom_coordinates=True)

        # Get test_orbit_id from the cluster detections
        test_orbit_id = cluster_detections.test_orbit_id[0].as_py()

        fitted_cluster = FittedClusters.from_kwargs(
            cluster_id=cluster.cluster_id,
            test_orbit_id=[test_orbit_id],
            time=gnomonic_coords.time[0],
            theta_x0=[x0],
            theta_y0=[y0],
            vtheta_x=[vx],
            vtheta_y=[vy],
            atheta_x=[ax],
            atheta_y=[ay],
            origin=gnomonic_coords.origin[0],
            frame=gnomonic_coords.frame,
            arc_length=[pc.subtract(pc.max(time), pc.min(time))],
            num_obs=[len(time)],
            chi2=[pc.sum(residuals.chi2)],
            rchi2=[pc.divide(pc.sum(residuals.chi2), pc.sum(residuals.dof).as_py() - 6)],
        )

        fitted_cluster_members = FittedClusterMembers.from_kwargs(
            cluster_id=cluster_members.cluster_id,
            obs_id=cluster_members.obs_id,
            test_orbit_id=pa.repeat(test_orbit_id, len(cluster_members)),
            residuals=residuals,
        )

        return fitted_cluster, fitted_cluster_members

    except np.linalg.LinAlgError:
        cluster_id = cluster.cluster_id[0].as_py()
        logger.warning(
            f"Failed to fit cluster {cluster_id}: Singular matrix (degenerate observations). Skipping cluster."
        )
        return FittedClusters.empty(), FittedClusterMembers.empty()
    except Exception as e:
        cluster_id = cluster.cluster_id[0].as_py()
        logger.warning(f"Failed to fit cluster {cluster_id}: {e}. Skipping cluster.")
        return FittedClusters.empty(), FittedClusterMembers.empty()


def fit_cluster_worker(
    clusters: Union[Clusters, ray.ObjectRef],
    cluster_members: Union[ClusterMembers, ray.ObjectRef],
    transformed_detections: Union[TransformedDetections, ray.ObjectRef],
    cluster_ids: List[str],
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Worker function for fitting clusters (used by Ray).

    This function selects clusters and their members by cluster_id,
    then calls fit_cluster for each.

    Parameters can be either the actual tables or Ray object references.
    """
    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()

    for cluster_id in cluster_ids:
        cluster_i = clusters.select("cluster_id", cluster_id)
        cluster_members_i = cluster_members.select("cluster_id", cluster_id)

        fitted_cluster_i, fitted_cluster_members_i = fit_cluster(
            cluster_i, cluster_members_i, transformed_detections
        )
        fitted_clusters = qv.concatenate([fitted_clusters, fitted_cluster_i])
        fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])

    return fitted_clusters, fitted_cluster_members


fit_cluster_worker_remote = ray.remote(fit_cluster_worker)
fit_cluster_worker_remote = fit_cluster_worker_remote.options(
    num_returns=1,
    num_cpus=1,
)


def fit_clusters(
    clusters: Clusters,
    cluster_members: ClusterMembers,
    transformed_detections: TransformedDetections,
    rchi2_threshold: Optional[float] = None,
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedClusters, FittedClusterMembers]:
    """
    Fit a set of clusters with a 2nd order polynomial motion model in theta_x and theta_y.

    Parameters
    ----------
    clusters : `~thor.clustering.data.Clusters`
        Clusters.
    cluster_members : `~thor.clustering.data.ClusterMembers`
        Cluster members.
    transformed_detections : `~thor.range_and_transform.TransformedDetections`
        Transformed detections.
    rchi2_threshold : float, optional
        Maximum reduced chi-squared value for a fitted cluster to be accepted.
        If None, no filtering is applied. [Default = None]
    chunk_size : int, optional
        Chunk size to use for parallelization.
    max_processes : int, optional
        Maximum number of processes to use for parallelization.
        [Default = 1]

    Returns
    -------
    fitted_clusters : `~thor.clustering.data.FittedClusters`
        Fitted clusters.
    fitted_cluster_members : `~thor.clustering.data.FittedClusterMembers`
        Fitted cluster members.
    """

    fitted_clusters = FittedClusters.empty()
    fitted_cluster_members = FittedClusterMembers.empty()
    if len(clusters) == 0:
        return fitted_clusters, fitted_cluster_members

    use_ray = initialize_use_ray(num_cpus=max_processes)
    cluster_ids = clusters.cluster_id.to_pylist()
    if use_ray:
        # Put tables in Ray object store to avoid repeated serialization
        clusters_ref = ray.put(clusters)
        cluster_members_ref = ray.put(cluster_members)
        if isinstance(transformed_detections, ray.ObjectRef):
            transformed_detections_ref = transformed_detections
        else:
            transformed_detections_ref = ray.put(transformed_detections)

        futures = []
        for cluster_id_chunk in _iterate_chunks(cluster_ids, chunk_size):
            futures.append(
                fit_cluster_worker_remote.remote(
                    clusters_ref, cluster_members_ref, transformed_detections_ref, cluster_id_chunk
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
                fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
                if fitted_clusters.fragmented():
                    fitted_clusters = qv.defragment(fitted_clusters)
                fitted_cluster_members = qv.concatenate(
                    [fitted_cluster_members, fitted_cluster_members_chunk]
                )
                if fitted_cluster_members.fragmented():
                    fitted_cluster_members = qv.defragment(fitted_cluster_members)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            fitted_clusters_chunk, fitted_cluster_members_chunk = ray.get(finished[0])
            fitted_clusters = qv.concatenate([fitted_clusters, fitted_clusters_chunk])
            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_chunk])
            if fitted_clusters.fragmented():
                fitted_clusters = qv.defragment(fitted_clusters)
            if fitted_cluster_members.fragmented():
                fitted_cluster_members = qv.defragment(fitted_cluster_members)
    else:
        for cluster_id in cluster_ids:
            cluster_i = clusters.select("cluster_id", cluster_id)
            cluster_members_i = cluster_members.select("cluster_id", cluster_id)
            fitted_cluster_i, fitted_cluster_members_i = fit_cluster(
                cluster_i, cluster_members_i, transformed_detections
            )
            fitted_clusters = qv.concatenate([fitted_clusters, fitted_cluster_i])
            fitted_cluster_members = qv.concatenate([fitted_cluster_members, fitted_cluster_members_i])

    # Filter by rchi2 threshold if specified
    if rchi2_threshold is not None and len(fitted_clusters) > 0:
        num_before = len(fitted_clusters)
        mask = pc.less_equal(fitted_clusters.rchi2, rchi2_threshold)
        fitted_clusters = fitted_clusters.apply_mask(mask)
        fitted_cluster_members = fitted_cluster_members.apply_mask(
            pc.is_in(fitted_cluster_members.cluster_id, fitted_clusters.cluster_id)
        )
        num_removed = num_before - len(fitted_clusters)
        if num_removed > 0:
            logger.info(f"Removed {num_removed} clusters with rchi2 > {rchi2_threshold}.")

    return fitted_clusters, fitted_cluster_members
