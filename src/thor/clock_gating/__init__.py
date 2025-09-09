"""
Kepler clock gating stage for THOR.

This module provides functionality to apply Kepler clock gating to observation-orbit
overlaps, producing K-chains (connected components) for downstream processing.
"""

from .apply_clock_gating import apply_clock_gating
from .kchains import Chains, ChainMembers

__all__ = ["apply_clock_gating", "Chains", "ChainMembers"]
