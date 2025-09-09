"""
Build BVH shards from TestOrbit objects.

This module provides a thin wrapper around adam_core's BVH sharding functionality,
adapting THOR's TestOrbit objects to adam_core's Orbits format.
"""

import tempfile
from pathlib import Path

from adam_core.geometry import build_bvh_shards, save_manifest

from ..orbit import TestOrbits


def build_from_test_orbits(
    test_orbits: TestOrbits,
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
    if len(test_orbits) == 0:
        raise ValueError("No test orbits provided")
    adam_orbits = test_orbits.to_orbits()
    
    # Build BVH shards using adam_core
    shards = build_bvh_shards(
        orbits=adam_orbits,
        max_chord_arcmin=max_chord_arcmin,
        target_shard_bytes=target_shard_bytes,
        float_dtype=float_dtype,
    )

    # Save manifest (adam-core saves to out_dir / 'manifest.json')
    save_manifest(
        out_dir=out_path,
        shards=shards,
        max_chord_arcmin=max_chord_arcmin,
        float_dtype=float_dtype,
    )

    return str(out_path / "manifest.json")
