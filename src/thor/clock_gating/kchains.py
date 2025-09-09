"""
K-chain data structures for THOR clock gating.

This module defines the data structures used to represent K-chains
(connected components of observations linked by orbital dynamics).
"""

import quivr as qv


class Chains(qv.Table):
    """
    K-chains metadata table.
    
    Each row represents a connected component (chain) of observations
    that are dynamically consistent with a test orbit.
    """
    chain_id = qv.LargeStringColumn()
    orbit_id = qv.LargeStringColumn()
    size = qv.Int32Column()
    t_min = qv.Float64Column()  # MJD
    t_max = qv.Float64Column()  # MJD


class ChainMembers(qv.Table):
    """
    K-chain membership table.
    
    Each row represents an observation that belongs to a specific chain.
    """
    chain_id = qv.LargeStringColumn()
    det_id = qv.LargeStringColumn()
    time_mjd = qv.Float64Column()
