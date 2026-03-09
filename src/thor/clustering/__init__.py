# ruff: noqa: F401
from .algorithms import ClusteringAlgorithm
from .data import (
    ClusterMembers,
    Clusters,
    FittedClusterMembers,
    FittedClusters,
    drop_duplicate_clusters,
    hash_obs_ids,
)
from .dbscan import VelocityGridDBSCAN
from .fft import VelocityGridFFT
from .fitting import fit_clusters
from .hotspot2d import VelocityGridHotspot2D
from .hough import HoughLineClustering
from .kdtree import VelocityGridKDTree
from .metrics import filter_clusters_by_length
from .optics import VelocityGridOPTICS
from .tiling import TileSpec, compute_auto_tile_size, compute_tile_grid, extract_tile_observations
from .tracklets import Tracklets, TrackletMembers, form_tracklets
from .velocity_grid import (
    VelocityGridBase,
    calculate_clustering_parameters_from_covariance,
)
from .windowing import TimeWindow, compute_linearity_window, compute_time_windows

try:
    from .cuda_shift_and_stack import CUDAShiftAndStack
except ImportError:
    pass

__all__ = [
    "ClusteringAlgorithm",
    "VelocityGridBase",
    "VelocityGridDBSCAN",
    "VelocityGridFFT",
    "VelocityGridHotspot2D",
    "VelocityGridKDTree",
    "VelocityGridOPTICS",
    "HoughLineClustering",
    "CUDAShiftAndStack",
    "Clusters",
    "ClusterMembers",
    "FittedClusters",
    "FittedClusterMembers",
    "drop_duplicate_clusters",
    "hash_obs_ids",
    "fit_clusters",
    "filter_clusters_by_length",
    "Tracklets",
    "TrackletMembers",
    "form_tracklets",
    "calculate_clustering_parameters_from_covariance",
    "TileSpec",
    "compute_tile_grid",
    "extract_tile_observations",
    "compute_auto_tile_size",
    "TimeWindow",
    "compute_linearity_window",
    "compute_time_windows",
]
