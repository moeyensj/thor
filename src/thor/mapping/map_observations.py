"""
Map observations to test orbits using BVH spatial indices.

This module provides a thin wrapper around adam_core's sharded BVH query functionality,
enabling efficient mapping of observation rays to overlapping test orbits.
"""

from adam_core.geometry import query_manifest_local, query_manifest_ray
from adam_core.rays import ObservationRays
from adam_core.geometry.overlap import OverlapHits


def map_observations_to_test_orbits(
    rays: ObservationRays,
    *,
    manifest_path: str,
    guard_arcmin: float = 1.0,
    alpha: float = 0.0,
    use_ray: bool = True,
    ray_batch_size: int = 200_000,
    max_concurrency: int | None = None,
) -> OverlapHits:
    """
    Map observation rays to overlapping test orbits using BVH spatial index.
    
    This function queries a sharded BVH manifest to find geometric overlaps
    between observation rays and test orbit segments.
    
    Parameters
    ----------
    rays : ObservationRays
        Observation rays to query against the BVH.
    manifest_path : str
        Path to the BVH manifest file.
    guard_arcmin : float, default=1.0
        Guard radius in arcminutes for geometric overlap detection.
    alpha : float, default=0.0
        Dynamic guard band parameter (0.0 = fixed guard).
    use_ray : bool, default=True
        Whether to use Ray for parallel processing.
    ray_batch_size : int, default=200_000
        Batch size for Ray parallel processing.
    max_concurrency : int, optional
        Maximum number of concurrent Ray tasks. If None, uses Ray defaults.
        
    Returns
    -------
    OverlapHits
        Table of geometric overlaps with columns:
        - det_id: Detection/observation ID
        - orbit_id: Test orbit ID
        - seg_id: Orbit segment ID
        - leaf_id: BVH leaf node ID
        - distance_au: Distance in AU
        
    Notes
    -----
    This function does not use THOR config - all parameters are explicit.
    Fixed-size padding is used internally to optimize JAX compilation.
    """
    if use_ray:
        return query_manifest_ray(
            rays=rays,
            manifest_path=manifest_path,
            guard_arcmin=guard_arcmin,
            alpha=alpha,
            ray_batch_size=ray_batch_size,
            max_concurrency=max_concurrency,
        )
    else:
        return query_manifest_local(
            rays=rays,
            manifest_path=manifest_path,
            guard_arcmin=guard_arcmin,
            alpha=alpha,
        )
