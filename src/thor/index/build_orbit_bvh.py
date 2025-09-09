"""
Build BVH shards from TestOrbit objects.

This module provides a thin wrapper around adam_core's BVH sharding functionality,
adapting THOR's TestOrbit objects to adam_core's Orbits format.
"""

import tempfile
from pathlib import Path
from typing import Iterable

from adam_core.geometry import build_bvh_shards, save_manifest

from ..orbit import TestOrbits


def build_from_test_orbits(
    test_orbits: Iterable[TestOrbits],
    *,
    out_dir: str | None = None,
    max_chord_arcmin: float = 60.0,
    target_shard_bytes: int = 3_000_000_000,
    float_dtype: str = "float64",
    overwrite: bool = False,
) -> str:
    """
    Build BVH shards from TestOrbit objects.
    
    This function converts TestOrbit objects to adam_core Orbits format and
    builds BVH shards for efficient geometric queries.
    
    Parameters
    ----------
    test_orbits : Iterable[TestOrbits]
        Test orbits to index in the BVH.
    out_dir : str, optional
        Output directory for shards and manifest. If None, uses a temporary directory.
    max_chord_arcmin : float, default=60.0
        Maximum chord length in arcminutes for orbit sampling.
    target_shard_bytes : int, default=3_000_000_000
        Target size in bytes for each shard.
    float_dtype : str, default="float64"
        Floating point precision for arrays.
    overwrite : bool, default=False
        Whether to overwrite existing files.
        
    Returns
    -------
    str
        Path to the manifest file.
        
    Notes
    -----
    This function does not use THOR config - all parameters are explicit.
    The caller is responsible for cleanup if out_dir is None (temp directory).
    """
    # Create output directory if needed
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="thor_bvh_")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Convert TestOrbits to adam_core Orbits
    # Concatenate all TestOrbits tables if multiple are provided
    if isinstance(test_orbits, TestOrbits):
        # Single table
        adam_orbits = test_orbits.to_orbits()
    else:
        # Multiple tables - concatenate them
        test_orbit_list = list(test_orbits)
        if not test_orbit_list:
            raise ValueError("No test orbits provided")
        
        if len(test_orbit_list) == 1:
            adam_orbits = test_orbit_list[0].to_orbits()
        else:
            # Concatenate multiple TestOrbits tables
            combined_test_orbits = TestOrbits.concat_tables(test_orbit_list)
            adam_orbits = combined_test_orbits.to_orbits()
    
    # Build BVH shards using adam_core
    shard_data_list = build_bvh_shards(
        orbits=adam_orbits,
        out_dir=str(out_path),
        max_chord_arcmin=max_chord_arcmin,
        target_shard_bytes=target_shard_bytes,
        float_dtype=float_dtype,
        overwrite=overwrite,
    )
    
    # Save manifest
    manifest_path = save_manifest(
        shard_data_list=shard_data_list,
        out_dir=str(out_path),
        overwrite=overwrite,
    )
    
    return manifest_path
