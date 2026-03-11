"""
CUDA-accelerated shift-and-stack clustering using atomic histograms.

This module implements the same shift-and-stack approach as
``gpu_shift_and_stack.py`` but replaces PyTorch with raw CUDA kernels
via CuPy, using a two-pass atomic histogram architecture:

1. **Coarse pass** (shared-memory atomics) — one thread block per velocity,
   bins observations into large spatial cells in shared memory.  Eliminates
   ~99 % of velocity candidates in microseconds.

2. **Fine pass** (global-memory atomics) — only for surviving candidates,
   bins into exact-size cells and counts per (velocity, bin) pair.

3. **Extract pass** — for hot (vel, bin) pairs, re-scans observations to
   collect member indices.

All three kernels fuse the velocity shift with binning, so no intermediate
shifted-coordinate arrays are ever materialised.

CuPy is an OPTIONAL dependency.  If not installed, importing this module
succeeds but instantiating ``CUDAShiftAndStack`` raises a clear error.
"""

import logging
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import ray

from ..orbit import TestOrbitEphemeris
from ..range_and_transform import TransformedDetections
from .data import ClusterMembers, Clusters, drop_duplicate_clusters
from .velocity_grid import calculate_clustering_parameters_from_covariance

logger = logging.getLogger("thor.clustering.cuda_shift_and_stack")

try:
    import cupy as cp
    from cupy import RawKernel

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Maximum shared memory per block we will request (bytes).
# 48 KiB is guaranteed on all CUDA architectures.
_MAX_SHARED_BYTES = 48 * 1024

# Block size for 1-D kernels (fine pass, extract).
_BLOCK_SIZE = 256

# Block size for coarse pass (threads per velocity block).
_COARSE_BLOCK_SIZE = 256

# -----------------------------------------------------------------------
# CUDA kernel sources
# -----------------------------------------------------------------------

_COARSE_COUNT_SRC = r"""
extern "C" __global__
void coarse_count(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ dt,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    int*         hot_flags,
    int          n_obs,
    float        inv_coarse_bin,
    float        x_origin,
    float        y_origin,
    int          n_bins_x,
    int          n_bins_y,
    int          min_obs
) {
    /*
     * One block per velocity.  Shared-memory histogram with coarse bins.
     * Sets hot_flags[vel] = 1 if any coarse bin reaches min_obs.
     */
    int vel = blockIdx.x;
    extern __shared__ int s_counts[];
    int n_spatial = n_bins_x * n_bins_y;

    // Zero shared histogram
    for (int i = threadIdx.x; i < n_spatial; i += blockDim.x)
        s_counts[i] = 0;
    __syncthreads();

    float vx_val = vx[vel];
    float vy_val = vy[vel];

    // Each thread loops over observations in strides of blockDim.x
    for (int obs = threadIdx.x; obs < n_obs; obs += blockDim.x) {
        float xs = x[obs] - vx_val * dt[obs];
        float ys = y[obs] - vy_val * dt[obs];
        int bx = __float2int_rd((xs - x_origin) * inv_coarse_bin);
        int by = __float2int_rd((ys - y_origin) * inv_coarse_bin);
        if (bx >= 0 && bx < n_bins_x && by >= 0 && by < n_bins_y) {
            atomicAdd(&s_counts[bx * n_bins_y + by], 1);
        }
    }
    __syncthreads();

    // Check for hot bins
    for (int i = threadIdx.x; i < n_spatial; i += blockDim.x) {
        if (s_counts[i] >= min_obs) {
            hot_flags[vel] = 1;
        }
    }
}
"""

_FINE_COUNT_SRC = r"""
extern "C" __global__
void fine_count(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ dt,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    int*         bin_counts,
    int          n_obs,
    int          n_candidates,
    float        inv_bin_size,
    float        x_origin,
    float        y_origin,
    int          n_bins_x,
    int          n_bins_y,
    int          n_spatial
) {
    /*
     * Grid: (n_candidates, ceil(n_obs / blockDim.x))
     * Each thread handles one (candidate_vel, observation) pair.
     * Atomically increments bin_counts[vel * n_spatial + flat_bin].
     */
    int vel = blockIdx.x;
    int obs = threadIdx.x + blockIdx.y * blockDim.x;
    if (vel >= n_candidates || obs >= n_obs) return;

    float xs = x[obs] - vx[vel] * dt[obs];
    float ys = y[obs] - vy[vel] * dt[obs];
    int bx = __float2int_rd((xs - x_origin) * inv_bin_size);
    int by = __float2int_rd((ys - y_origin) * inv_bin_size);

    if (bx >= 0 && bx < n_bins_x && by >= 0 && by < n_bins_y) {
        int flat = bx * n_bins_y + by;
        atomicAdd(&bin_counts[vel * n_spatial + flat], 1);
    }
}
"""

_EXTRACT_MEMBERS_SRC = r"""
extern "C" __global__
void extract_members(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ dt,
    const float* __restrict__ hot_vx,
    const float* __restrict__ hot_vy,
    const int*   __restrict__ hot_bins,
    int*         member_counts,
    int*         member_obs,
    int          n_obs,
    int          n_hot,
    int          max_members,
    float        inv_bin_size,
    float        x_origin,
    float        y_origin,
    int          n_bins_y
) {
    /*
     * Grid: (n_hot, ceil(n_obs / blockDim.x))
     * Each thread checks whether observation `obs` lands in the
     * hot bin for hot pair `hot_idx`.  If so, stores the obs index.
     */
    int hot_idx = blockIdx.x;
    int obs = threadIdx.x + blockIdx.y * blockDim.x;
    if (hot_idx >= n_hot || obs >= n_obs) return;

    float xs = x[obs] - hot_vx[hot_idx] * dt[obs];
    float ys = y[obs] - hot_vy[hot_idx] * dt[obs];
    int bx = __float2int_rd((xs - x_origin) * inv_bin_size);
    int by = __float2int_rd((ys - y_origin) * inv_bin_size);
    int flat = bx * n_bins_y + by;

    if (flat == hot_bins[hot_idx]) {
        int pos = atomicAdd(&member_counts[hot_idx], 1);
        if (pos < max_members) {
            member_obs[hot_idx * max_members + pos] = obs;
        }
    }
}
"""


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _compile_kernels():
    """Compile and cache the three CUDA kernels."""
    coarse = RawKernel(_COARSE_COUNT_SRC, "coarse_count")
    fine = RawKernel(_FINE_COUNT_SRC, "fine_count")
    extract = RawKernel(_EXTRACT_MEMBERS_SRC, "extract_members")
    return coarse, fine, extract


def _shifted_bounds(
    x: np.ndarray,
    y: np.ndarray,
    dt: np.ndarray,
    vx_chunk: np.ndarray,
    vy_chunk: np.ndarray,
    ox: float,
    oy: float,
) -> Tuple[float, float, float, float]:
    """
    Compute conservative (x_min, x_max, y_min, y_max) of shifted
    coordinates across all velocities in the chunk and all observations.
    """
    # shifted_x = (x + ox) - vx * dt
    # For each obs, extremes over vx range are at vx_min and vx_max.
    x_off = x + ox
    y_off = y + oy
    vx_lo, vx_hi = float(vx_chunk.min()), float(vx_chunk.max())
    vy_lo, vy_hi = float(vy_chunk.min()), float(vy_chunk.max())

    sx_at_lo = x_off - vx_lo * dt
    sx_at_hi = x_off - vx_hi * dt
    x_min = min(float(sx_at_lo.min()), float(sx_at_hi.min()))
    x_max = max(float(sx_at_lo.max()), float(sx_at_hi.max()))

    sy_at_lo = y_off - vy_lo * dt
    sy_at_hi = y_off - vy_hi * dt
    y_min = min(float(sy_at_lo.min()), float(sy_at_hi.min()))
    y_max = max(float(sy_at_lo.max()), float(sy_at_hi.max()))

    return x_min, x_max, y_min, y_max


def _grid_dims(
    extent_min: float,
    extent_max: float,
    bin_size: float,
) -> int:
    """Number of bins to cover [extent_min, extent_max] with given bin_size."""
    return max(int(np.ceil((extent_max - extent_min) / bin_size)) + 1, 1)


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------


class CUDAShiftAndStack:
    """
    CUDA-accelerated shift-and-stack clustering using atomic histograms.

    Implements the ``ClusteringAlgorithm`` protocol.  Uses a three-pass
    architecture (coarse -> fine -> extract) with fused shift+bin kernels
    so no intermediate shifted-coordinate arrays are ever materialised.

    Requires CuPy.  Falls back with a clear error if unavailable.
    """

    def __init__(
        self,
        radius: float = 0.005,
        min_obs: int = 6,
        min_arc_length: float = 1.0,
        min_nights: int = 3,
        vx_range: Optional[List[float]] = None,
        vy_range: Optional[List[float]] = None,
        vx_bins: Optional[int] = None,
        vy_bins: Optional[int] = None,
        vx_values: Optional[npt.NDArray[np.float64]] = None,
        vy_values: Optional[npt.NDArray[np.float64]] = None,
        velocity_bin_separation: float = 2.0,
        mahalanobis_distance: Optional[float] = None,
        radius_multiplier: float = 1.5,
        density_multiplier: float = 2.5,
        min_radius: float = 0.01 / 3600,
        max_radius: float = 0.05,
        astrometric_precision: Optional[float] = None,
        max_bins: int = 5000,
        coarse_factor: int = 5,
        chunk_size: Optional[int] = None,
        max_processes: Optional[int] = 1,
        whiten: bool = False,
        device: Optional[int] = None,
    ):
        if not CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is required for CUDAShiftAndStack but is not installed. "
                "Install it with: pip install cupy-cuda12x  "
                "(see https://docs.cupy.dev/en/stable/install.html)"
            )

        self.radius = radius
        self.min_obs = min_obs
        self.min_arc_length = min_arc_length
        self.min_nights = min_nights
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vx_bins = vx_bins
        self.vy_bins = vy_bins
        self.vx_values = vx_values
        self.vy_values = vy_values
        self.velocity_bin_separation = velocity_bin_separation
        self.mahalanobis_distance = mahalanobis_distance
        self.radius_multiplier = radius_multiplier
        self.density_multiplier = density_multiplier
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.astrometric_precision = astrometric_precision
        self.max_bins = max_bins
        self.coarse_factor = coarse_factor
        self.chunk_size = chunk_size
        self.max_processes = max_processes
        self.whiten = whiten

        if device is not None:
            self._device_id = device
        else:
            self._device_id = 0

        cp.cuda.Device(self._device_id).use()
        logger.info(f"CUDAShiftAndStack using CUDA device {self._device_id}")

        # Compile kernels (cached by CuPy after first call)
        self._k_coarse, self._k_fine, self._k_extract = _compile_kernels()

    # -----------------------------------------------------------------
    # Velocity grid resolution (mirrors GPUShiftAndStack)
    # -----------------------------------------------------------------

    def _resolve_velocity_grid(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        radius = self.radius

        if self.vx_values is not None and self.vy_values is not None:
            if len(self.vx_values) != len(self.vy_values):
                raise ValueError(
                    f"vx_values and vy_values must have same length. "
                    f"Got {len(self.vx_values)} and {len(self.vy_values)}."
                )
            return self.vx_values, self.vy_values, radius

        if test_orbit_ephemeris is not None:
            try:
                vxx, vyy, cov_radius, metadata = calculate_clustering_parameters_from_covariance(
                    test_orbit_ephemeris,
                    transformed_detections,
                    mahalanobis_distance=(
                        self.mahalanobis_distance if self.mahalanobis_distance is not None else 3.0
                    ),
                    velocity_bin_separation=self.velocity_bin_separation,
                    radius_multiplier=self.radius_multiplier,
                    density_multiplier=self.density_multiplier,
                    min_radius=self.min_radius,
                    max_radius=self.max_radius,
                    whiten=self.whiten,
                    astrometric_precision=self.astrometric_precision,
                    max_bins=self.max_bins,
                )
                radius = cov_radius
                logger.info(
                    f"Covariance-informed clustering: radius={radius:.6f} deg "
                    f"({radius * 3600:.3f} arcsec), {len(vxx)} velocity grid points"
                )
                return vxx, vyy, radius
            except Exception as e:
                logger.warning(f"Failed to calculate covariance parameters: {e}. Using defaults.")

        vx_range = self.vx_range if self.vx_range is not None else [-0.1, 0.1]
        vy_range = self.vy_range if self.vy_range is not None else [-0.1, 0.1]
        vx_n = self.vx_bins if self.vx_bins is not None else 100
        vy_n = self.vy_bins if self.vy_bins is not None else 100
        vx = np.linspace(*vx_range, num=vx_n)
        vy = np.linspace(*vy_range, num=vy_n)
        vxx, vyy = np.meshgrid(vx, vy)
        return vxx.flatten(), vyy.flatten(), radius

    # -----------------------------------------------------------------
    # Auto chunk size
    # -----------------------------------------------------------------

    def _auto_chunk_size(self, n_obs: int, n_spatial_fine: int) -> int:
        """
        Choose how many candidate velocities to process in the fine
        pass per iteration, based on available GPU memory.

        Fine-pass memory: n_candidates * n_spatial_fine * 4 bytes.
        """
        if self.chunk_size is not None:
            return self.chunk_size

        mem_free = cp.cuda.Device(self._device_id).mem_info[0]
        usable = int(mem_free * 0.6)
        bytes_per_vel = n_spatial_fine * 4  # int32 histogram per velocity
        max_chunk = max(1, usable // max(bytes_per_vel, 1))
        chunk = min(max_chunk, 50_000)
        logger.info(
            f"Fine-pass auto chunk_size={chunk} "
            f"(n_spatial={n_spatial_fine}, usable={usable / 1e9:.2f} GB)"
        )
        return chunk

    # -----------------------------------------------------------------
    # Coarse pass
    # -----------------------------------------------------------------

    def _coarse_pass(
        self,
        x_d: "cp.ndarray",
        y_d: "cp.ndarray",
        dt_d: "cp.ndarray",
        vx_d: "cp.ndarray",
        vy_d: "cp.ndarray",
        n_obs: int,
        n_vel: int,
        coarse_bin_size: float,
        x_origin: float,
        y_origin: float,
        x_max: float,
        y_max: float,
    ) -> npt.NDArray[np.intp]:
        """
        Run the coarse shared-memory pass.  Returns numpy array of
        candidate velocity indices.
        """
        n_bins_x = _grid_dims(x_origin, x_max, coarse_bin_size)
        n_bins_y = _grid_dims(y_origin, y_max, coarse_bin_size)
        n_coarse_spatial = n_bins_x * n_bins_y
        shared_bytes = n_coarse_spatial * 4  # int32 per bin

        if shared_bytes > _MAX_SHARED_BYTES:
            # Coarse grid too large for shared memory — skip prefilter
            logger.info(
                f"Coarse grid {n_bins_x}x{n_bins_y} = {n_coarse_spatial} bins "
                f"exceeds shared memory ({shared_bytes / 1024:.0f} KB > "
                f"{_MAX_SHARED_BYTES / 1024:.0f} KB). Skipping coarse prefilter."
            )
            return np.arange(n_vel)

        hot_flags = cp.zeros(n_vel, dtype=cp.int32)

        self._k_coarse(
            (n_vel,),                    # grid: one block per velocity
            (_COARSE_BLOCK_SIZE,),       # block
            (
                x_d, y_d, dt_d, vx_d, vy_d,
                hot_flags,
                np.int32(n_obs),
                np.float32(1.0 / coarse_bin_size),
                np.float32(x_origin),
                np.float32(y_origin),
                np.int32(n_bins_x),
                np.int32(n_bins_y),
                np.int32(self.min_obs),
            ),
            shared_mem=shared_bytes,
        )

        candidates = cp.nonzero(hot_flags)[0].get()
        return candidates

    # -----------------------------------------------------------------
    # Fine pass
    # -----------------------------------------------------------------

    def _fine_pass(
        self,
        x_d: "cp.ndarray",
        y_d: "cp.ndarray",
        dt_d: "cp.ndarray",
        cand_vx_d: "cp.ndarray",
        cand_vy_d: "cp.ndarray",
        n_obs: int,
        n_candidates: int,
        bin_size: float,
        x_origin: float,
        y_origin: float,
        x_max: float,
        y_max: float,
    ) -> Tuple[npt.NDArray[np.int32], int, int, int]:
        """
        Run the fine global-memory pass for candidate velocities.
        Returns (bin_counts_host, n_bins_x, n_bins_y, n_spatial).
        """
        n_bins_x = _grid_dims(x_origin, x_max, bin_size)
        n_bins_y = _grid_dims(y_origin, y_max, bin_size)
        n_spatial = n_bins_x * n_bins_y

        bin_counts = cp.zeros(n_candidates * n_spatial, dtype=cp.int32)

        obs_blocks = (n_obs + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        self._k_fine(
            (n_candidates, obs_blocks),  # grid
            (_BLOCK_SIZE,),              # block
            (
                x_d, y_d, dt_d, cand_vx_d, cand_vy_d,
                bin_counts,
                np.int32(n_obs),
                np.int32(n_candidates),
                np.float32(1.0 / bin_size),
                np.float32(x_origin),
                np.float32(y_origin),
                np.int32(n_bins_x),
                np.int32(n_bins_y),
                np.int32(n_spatial),
            ),
        )

        bin_counts_host = bin_counts.get()
        del bin_counts
        return bin_counts_host, n_bins_x, n_bins_y, n_spatial

    # -----------------------------------------------------------------
    # Extract pass
    # -----------------------------------------------------------------

    def _extract_pass(
        self,
        x_d: "cp.ndarray",
        y_d: "cp.ndarray",
        dt_d: "cp.ndarray",
        hot_vx: np.ndarray,
        hot_vy: np.ndarray,
        hot_bins: np.ndarray,
        hot_counts: np.ndarray,
        n_obs: int,
        bin_size: float,
        x_origin: float,
        y_origin: float,
        n_bins_y: int,
    ) -> List[npt.NDArray[np.intp]]:
        """
        For each hot (vel, bin) pair, collect observation indices.
        Returns list of arrays of observation indices.
        """
        n_hot = len(hot_vx)
        if n_hot == 0:
            return []

        max_members = int(hot_counts.max()) + 1

        hot_vx_d = cp.asarray(hot_vx, dtype=cp.float32)
        hot_vy_d = cp.asarray(hot_vy, dtype=cp.float32)
        hot_bins_d = cp.asarray(hot_bins, dtype=cp.int32)
        member_counts_d = cp.zeros(n_hot, dtype=cp.int32)
        member_obs_d = cp.full(n_hot * max_members, -1, dtype=cp.int32)

        obs_blocks = (n_obs + _BLOCK_SIZE - 1) // _BLOCK_SIZE
        self._k_extract(
            (n_hot, obs_blocks),
            (_BLOCK_SIZE,),
            (
                x_d, y_d, dt_d,
                hot_vx_d, hot_vy_d, hot_bins_d,
                member_counts_d, member_obs_d,
                np.int32(n_obs),
                np.int32(n_hot),
                np.int32(max_members),
                np.float32(1.0 / bin_size),
                np.float32(x_origin),
                np.float32(y_origin),
                np.int32(n_bins_y),
            ),
        )

        member_counts_h = member_counts_d.get()
        member_obs_h = member_obs_d.reshape(n_hot, max_members).get()

        del hot_vx_d, hot_vy_d, hot_bins_d, member_counts_d, member_obs_d

        results = []
        for i in range(n_hot):
            count = member_counts_h[i]
            members = member_obs_h[i, :count]
            results.append(members.astype(np.intp))

        return results

    # -----------------------------------------------------------------
    # Cluster building + temporal filtering
    # -----------------------------------------------------------------

    def _build_clusters_from_members(
        self,
        member_lists: List[npt.NDArray[np.intp]],
        hot_vel_global_indices: np.ndarray,
        vxx: np.ndarray,
        vyy: np.ndarray,
        dt: np.ndarray,
        nights: np.ndarray,
        obs_ids: np.ndarray,
        # Accumulators (modified in place)
        all_cluster_ids: List[str],
        all_cluster_vx: List[float],
        all_cluster_vy: List[float],
        all_cluster_arc_lengths: List[float],
        all_cluster_num_obs: List[int],
        all_member_cluster_ids: List[str],
        all_member_obs_ids: List[str],
    ) -> int:
        """Apply temporal filters and accumulate cluster records."""
        n_added = 0
        for i, member_indices in enumerate(member_lists):
            if len(member_indices) == 0:
                continue

            dt_c = dt[member_indices]

            # Keep one observation per unique timestamp
            _, unique_idx = np.unique(dt_c, return_index=True)
            member_indices = member_indices[unique_idx]
            dt_c = dt_c[unique_idx]

            if len(member_indices) < self.min_obs:
                continue

            arc_length = float(dt_c.max() - dt_c.min())
            if arc_length < self.min_arc_length:
                continue

            nights_c = nights[member_indices]
            if len(np.unique(nights_c)) < self.min_nights:
                continue

            v_global = hot_vel_global_indices[i]
            cid = uuid.uuid4().hex
            member_obs = obs_ids[member_indices]

            all_cluster_ids.append(cid)
            all_cluster_vx.append(float(vxx[v_global]))
            all_cluster_vy.append(float(vyy[v_global]))
            all_cluster_arc_lengths.append(arc_length)
            all_cluster_num_obs.append(len(member_obs))
            all_member_cluster_ids.extend([cid] * len(member_obs))
            all_member_obs_ids.extend(member_obs.tolist())
            n_added += 1

        return n_added

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def find_clusters(
        self,
        transformed_detections: TransformedDetections,
        test_orbit_ephemeris: Optional[TestOrbitEphemeris] = None,
        tracklets=None,
        tracklet_members=None,
    ) -> Tuple[Clusters, ClusterMembers]:
        time_start = time.perf_counter()
        logger.info("Running CUDA atomic-histogram shift-and-stack clustering...")

        # Handle Ray ObjectRef inputs
        if isinstance(transformed_detections, ray.ObjectRef):
            transformed_detections = ray.get(transformed_detections)
        if isinstance(test_orbit_ephemeris, ray.ObjectRef):
            test_orbit_ephemeris = ray.get(test_orbit_ephemeris)

        if len(transformed_detections) == 0:
            logger.info("No observations to cluster.")
            return Clusters.empty(), ClusterMembers.empty()

        unique_times = transformed_detections.coordinates.time.unique()
        if len(unique_times) < self.min_obs:
            logger.info("Not enough unique observation times.")
            return Clusters.empty(), ClusterMembers.empty()

        time_range = unique_times.max().mjd()[0].as_py() - unique_times.min().mjd()[0].as_py()
        if time_range < self.min_arc_length:
            logger.info("Time range less than minimum arc length.")
            return Clusters.empty(), ClusterMembers.empty()

        # Resolve velocity grid and radius
        vxx, vyy, radius = self._resolve_velocity_grid(
            transformed_detections, test_orbit_ephemeris
        )
        bin_size = 2.0 * radius
        coarse_bin_size = bin_size * self.coarse_factor

        # Extract observation arrays
        obs_ids = transformed_detections.id.to_numpy(zero_copy_only=False)
        nights = transformed_detections.night.to_numpy(zero_copy_only=False)
        x = transformed_detections.coordinates.theta_x.to_numpy(zero_copy_only=False).astype(
            np.float32
        )
        y = transformed_detections.coordinates.theta_y.to_numpy(zero_copy_only=False).astype(
            np.float32
        )
        mjd = transformed_detections.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        dt = (mjd - mjd.min()).astype(np.float32)

        # Drop NaNs
        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dt)
        if not np.all(finite_mask):
            n_drop = np.size(finite_mask) - np.count_nonzero(finite_mask)
            logger.warning(f"Dropping {n_drop} observations with NaN coordinates.")
            x, y, dt = x[finite_mask], y[finite_mask], dt[finite_mask]
            nights, obs_ids = nights[finite_mask], obs_ids[finite_mask]

        n_obs = len(x)
        if n_obs < self.min_obs:
            logger.info(f"Only {n_obs} finite observations, fewer than min_obs={self.min_obs}.")
            return Clusters.empty(), ClusterMembers.empty()

        n_vel = len(vxx)

        logger.info(f"Clustering radius: {radius:.6f} deg ({radius * 3600:.3f} arcsec)")
        logger.info(
            f"Bin size: {bin_size:.6f} deg, coarse bin size: {coarse_bin_size:.6f} deg"
        )
        logger.info(f"CUDA shift-and-stack: {n_vel} vel x {n_obs} obs")

        # Upload observations to GPU (once, reused for all offsets and chunks)
        cp.cuda.Device(self._device_id).use()
        x_d = cp.asarray(x)
        y_d = cp.asarray(y)
        dt_d = cp.asarray(dt)

        offsets = [
            (0.0, 0.0),
            (radius, 0.0),
            (0.0, radius),
            (radius, radius),
        ]

        # Accumulators
        all_cluster_ids: List[str] = []
        all_cluster_vx: List[float] = []
        all_cluster_vy: List[float] = []
        all_cluster_arc_lengths: List[float] = []
        all_cluster_num_obs: List[int] = []
        all_member_cluster_ids: List[str] = []
        all_member_obs_ids: List[str] = []

        time_kernel = time.perf_counter()
        total_raw = 0
        total_coarse_survived = 0

        for ox, oy in offsets:
            # Compute conservative bounds for shifted coordinates
            # _shifted_bounds returns bounds of (x + ox) - vx * dt, but kernels
            # compute x - vx * dt (no offset).  Subtracting offset from origin
            # and max makes the kernel's binning equivalent:
            #   floor((x - vx*dt - (origin - ox)) / bin) = floor(((x+ox) - vx*dt - origin) / bin)
            x_origin, x_max, y_origin, y_max = _shifted_bounds(
                x, y, dt, vxx, vyy, ox, oy
            )
            kx_origin = x_origin - ox
            kx_max = x_max - ox
            ky_origin = y_origin - oy
            ky_max = y_max - oy

            # Coarse pass: upload full velocity arrays
            vx_d = cp.asarray(vxx.astype(np.float32))
            vy_d = cp.asarray(vyy.astype(np.float32))

            candidates = self._coarse_pass(
                x_d, y_d, dt_d, vx_d, vy_d,
                n_obs, n_vel,
                coarse_bin_size,
                kx_origin, ky_origin, kx_max, ky_max,
            )

            del vx_d, vy_d
            n_candidates = len(candidates)
            total_coarse_survived += n_candidates

            if n_candidates == 0:
                continue

            logger.info(
                f"  Offset ({ox:.4f}, {oy:.4f}): "
                f"{n_candidates}/{n_vel} candidates after coarse pass "
                f"({100 * n_candidates / n_vel:.1f}%)"
            )

            # Fine pass: process candidates in chunks
            cand_vx = vxx[candidates].astype(np.float32)
            cand_vy = vyy[candidates].astype(np.float32)

            # Recompute tighter bounds for candidates only (kernel-adjusted)
            x_origin_f, x_max_f, y_origin_f, y_max_f = _shifted_bounds(
                x, y, dt, cand_vx, cand_vy, ox, oy
            )
            kx_origin_f = x_origin_f - ox
            kx_max_f = x_max_f - ox
            ky_origin_f = y_origin_f - oy
            ky_max_f = y_max_f - oy
            # Grid dims use the range which is the same with or without offset
            n_bins_x_f = _grid_dims(kx_origin_f, kx_max_f, bin_size)
            n_bins_y_f = _grid_dims(ky_origin_f, ky_max_f, bin_size)
            n_spatial_f = n_bins_x_f * n_bins_y_f

            fine_chunk = self._auto_chunk_size(n_obs, n_spatial_f)

            for chunk_start in range(0, n_candidates, fine_chunk):
                chunk_end = min(chunk_start + fine_chunk, n_candidates)
                chunk_vx = cand_vx[chunk_start:chunk_end]
                chunk_vy = cand_vy[chunk_start:chunk_end]
                n_chunk = chunk_end - chunk_start

                chunk_vx_d = cp.asarray(chunk_vx)
                chunk_vy_d = cp.asarray(chunk_vy)

                bin_counts_h, _, n_bins_y_fine, n_spatial = self._fine_pass(
                    x_d, y_d, dt_d,
                    chunk_vx_d, chunk_vy_d,
                    n_obs, n_chunk,
                    bin_size,
                    kx_origin_f, ky_origin_f, kx_max_f, ky_max_f,
                )

                del chunk_vx_d, chunk_vy_d

                # Find hot (velocity, bin) pairs
                bin_counts_2d = bin_counts_h.reshape(n_chunk, n_spatial)
                hot_vel_local, hot_bin_flat = np.nonzero(bin_counts_2d >= self.min_obs)

                if len(hot_vel_local) == 0:
                    continue

                hot_vx_np = chunk_vx[hot_vel_local]
                hot_vy_np = chunk_vy[hot_vel_local]
                hot_bins_np = hot_bin_flat.astype(np.int32)
                hot_counts_np = bin_counts_2d[hot_vel_local, hot_bin_flat]

                # Extract members for hot pairs
                member_lists = self._extract_pass(
                    x_d, y_d, dt_d,
                    hot_vx_np, hot_vy_np, hot_bins_np, hot_counts_np,
                    n_obs, bin_size,
                    kx_origin_f, ky_origin_f, n_bins_y_fine,
                )

                # Map local candidate indices back to global velocity indices
                hot_vel_global = candidates[chunk_start + hot_vel_local]

                n_added = self._build_clusters_from_members(
                    member_lists, hot_vel_global, vxx, vyy,
                    dt, nights, obs_ids,
                    all_cluster_ids, all_cluster_vx, all_cluster_vy,
                    all_cluster_arc_lengths, all_cluster_num_obs,
                    all_member_cluster_ids, all_member_obs_ids,
                )
                total_raw += n_added

        time_kernel_end = time.perf_counter()

        # Free GPU memory
        del x_d, y_d, dt_d
        cp.get_default_memory_pool().free_all_blocks()

        coarse_total = n_vel * len(offsets)
        logger.info(
            f"CUDA kernels: {n_vel} vel x {len(offsets)} offsets "
            f"in {time_kernel_end - time_kernel:.3f}s"
        )
        logger.info(
            f"Coarse prefilter: {total_coarse_survived}/{coarse_total} survived "
            f"({100 * total_coarse_survived / max(coarse_total, 1):.1f}%)"
        )
        logger.info(f"Raw clusters from fine pass: {total_raw}")

        if len(all_cluster_ids) == 0:
            logger.info("Found 0 clusters.")
            logger.info(
                f"CUDA shift-and-stack completed in "
                f"{time.perf_counter() - time_start:.3f}s"
            )
            return Clusters.empty(), ClusterMembers.empty()

        # Build quivr tables
        all_clusters = Clusters.from_kwargs(
            cluster_id=all_cluster_ids,
            vtheta_x=np.array(all_cluster_vx),
            vtheta_y=np.array(all_cluster_vy),
            arc_length=all_cluster_arc_lengths,
            num_obs=all_cluster_num_obs,
        )
        all_cluster_members = ClusterMembers.from_kwargs(
            cluster_id=all_member_cluster_ids,
            obs_id=all_member_obs_ids,
        )

        # Deduplicate
        all_clusters = all_clusters.sort_by([("cluster_id", "ascending")])
        all_cluster_members = all_cluster_members.sort_by([("cluster_id", "ascending")])
        num_before = len(all_clusters)
        all_clusters, all_cluster_members = drop_duplicate_clusters(
            all_clusters, all_cluster_members
        )
        logger.info(f"Removed {num_before - len(all_clusters)} duplicate clusters.")
        logger.info(f"Found {len(all_clusters)} clusters.")
        logger.info(
            f"CUDA shift-and-stack completed in "
            f"{time.perf_counter() - time_start:.3f}s"
        )

        return all_clusters, all_cluster_members
