"""
Apply Kepler clock gating to observation-orbit overlaps.

This module provides functionality to filter observation-orbit overlaps
using Kepler's laws and produce K-chains for downstream processing.
"""

import uuid
from typing import Dict, Any, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from adam_core.geometry.overlap import OverlapHits
from adam_core.rays import ObservationRays

from .kchains import Chains, ChainMembers


def apply_clock_gating(
    hits: OverlapHits,
    rays: ObservationRays,
    cfg: Dict[str, Any] | None = None,
) -> Tuple[Chains, ChainMembers]:
    """
    Apply Kepler clock gating to observation-orbit overlaps.
    
    This is currently a stub implementation that produces trivial K-chains
    without actual clock gating. Each orbit gets one chain containing all
    its associated detections.
    
    Parameters
    ----------
    hits : OverlapHits
        Geometric overlaps between observations and test orbits.
    rays : ObservationRays
        Observation rays containing timing information.
    cfg : dict, optional
        Clock gating configuration parameters. Currently unused in stub.
        
    Returns
    -------
    chains : Chains
        K-chains metadata table.
    chain_members : ChainMembers
        K-chain membership table.
        
    Notes
    -----
    This stub implementation creates one chain per orbit containing all
    detections associated with that orbit. Future implementation will
    apply actual Kepler clock gating using JAX kernels.
    """
    if len(hits) == 0:
        # Return empty tables
        return (
            Chains.empty(),
            ChainMembers.empty(),
        )
    
    # Get unique orbit IDs from hits
    unique_orbit_ids = pc.unique(hits.orbit_id.to_pyarrow()).to_pylist()
    
    # Create a lookup table for ray times
    ray_times_dict = {}
    for i in range(len(rays)):
        det_id = rays.det_id[i].as_py()
        time_mjd = rays.coordinates.time[i].mjd().as_py()
        ray_times_dict[det_id] = time_mjd
    
    # Build chains - one per orbit
    chain_data = {
        "chain_id": [],
        "orbit_id": [],
        "size": [],
        "t_min": [],
        "t_max": [],
    }
    
    member_data = {
        "chain_id": [],
        "det_id": [],
        "time_mjd": [],
    }
    
    for orbit_id in unique_orbit_ids:
        # Find all detections for this orbit
        orbit_mask = pc.equal(hits.orbit_id.to_pyarrow(), orbit_id)
        orbit_det_ids = pc.filter(hits.det_id.to_pyarrow(), orbit_mask).to_pylist()
        
        # Get unique detection IDs (in case of multiple segments per detection)
        unique_det_ids = list(set(orbit_det_ids))
        
        if not unique_det_ids:
            continue
            
        # Get times for these detections
        det_times = []
        for det_id in unique_det_ids:
            if det_id in ray_times_dict:
                det_times.append(ray_times_dict[det_id])
        
        if not det_times:
            continue
            
        # Create chain
        chain_id = f"{orbit_id}_{uuid.uuid4().hex[:8]}"
        
        chain_data["chain_id"].append(chain_id)
        chain_data["orbit_id"].append(orbit_id)
        chain_data["size"].append(len(unique_det_ids))
        chain_data["t_min"].append(min(det_times))
        chain_data["t_max"].append(max(det_times))
        
        # Add members
        for det_id in unique_det_ids:
            if det_id in ray_times_dict:
                member_data["chain_id"].append(chain_id)
                member_data["det_id"].append(det_id)
                member_data["time_mjd"].append(ray_times_dict[det_id])
    
    # Create tables
    chains = Chains.from_kwargs(**chain_data)
    chain_members = ChainMembers.from_kwargs(**member_data)
    
    return chains, chain_members
