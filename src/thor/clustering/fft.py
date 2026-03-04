import logging
from typing import Callable, List

import numpy as np
import numpy.typing as npt
import ray
from scipy import ndimage
from scipy.signal import fftconvolve

from .velocity_grid import VelocityGridBase, _cluster_velocity_find_worker

logger = logging.getLogger(__name__)


def _find_clusters_fft(
    points: npt.NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> List[npt.NDArray[np.int64]]:
    """
    Find clusters using FFT-based density peak detection.

    Rasterizes the point cloud into a 2D histogram, convolves with a
    circular top-hat kernel of radius ``eps``, and finds connected
    regions above a ``min_samples`` threshold. Points within ``eps``
    of a peak region center are assigned to that cluster.

    Parameters
    ----------
    points : `~numpy.ndarray` (N, 2)
        2D array of point coordinates.
    eps : float
        Radius for the circular convolution kernel.
    min_samples : int
        Minimum density (count) threshold for peak detection.

    Returns
    -------
    clusters : list of `~numpy.ndarray`
        List of arrays, each containing the indices of points
        belonging to a cluster.
    """
    n = len(points)
    if n == 0:
        return []

    x = points[:, 0]
    y = points[:, 1]

    # Grid cell size ~ eps/2 (Nyquist sampling)
    cell_size = eps / 2.0
    if cell_size <= 0:
        return []

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Add padding of eps around the data extent
    x_min -= eps
    x_max += eps
    y_min -= eps
    y_max += eps

    nx = max(int(np.ceil((x_max - x_min) / cell_size)), 1)
    ny = max(int(np.ceil((y_max - y_min) / cell_size)), 1)

    # Cap grid size to avoid excessive memory usage
    max_grid = 4096
    if nx > max_grid or ny > max_grid:
        scale = max(nx, ny) / max_grid
        cell_size *= scale
        nx = max(int(np.ceil((x_max - x_min) / cell_size)), 1)
        ny = max(int(np.ceil((y_max - y_min) / cell_size)), 1)

    # Rasterize points into 2D histogram
    ix = np.clip(((x - x_min) / cell_size).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / cell_size).astype(int), 0, ny - 1)
    histogram = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(histogram, (ix, iy), 1)

    # Build circular top-hat kernel
    kernel_radius_cells = max(int(np.ceil(eps / cell_size)), 1)
    ky, kx = np.mgrid[
        -kernel_radius_cells : kernel_radius_cells + 1, -kernel_radius_cells : kernel_radius_cells + 1
    ]
    kernel = ((kx**2 + ky**2) <= kernel_radius_cells**2).astype(np.float64)

    # Convolve using FFT
    density = fftconvolve(histogram, kernel, mode="same")

    # Threshold to find peak regions
    peak_mask = density >= min_samples

    if not np.any(peak_mask):
        return []

    # Label connected peak regions
    labeled, num_labels = ndimage.label(peak_mask)

    if num_labels == 0:
        return []

    # Compute center of mass for each labeled region (in grid coords)
    region_centers = ndimage.center_of_mass(density, labeled, range(1, num_labels + 1))

    # Convert region centers back to data coordinates
    clusters = []
    for cx, cy in region_centers:
        center_x = x_min + cx * cell_size
        center_y = y_min + cy * cell_size

        # Assign points within eps of this center
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        member_mask = dist_sq <= eps**2
        indices = np.where(member_mask)[0]

        if len(indices) >= min_samples:
            clusters.append(indices)

    # Resolve overlapping assignments: each point goes to the nearest center
    if len(clusters) > 1:
        centers = np.array([[x_min + cx * cell_size, y_min + cy * cell_size] for cx, cy in region_centers])
        assignment = np.full(n, -1, dtype=np.intp)
        for ci, indices in enumerate(clusters):
            for idx in indices:
                if assignment[idx] == -1:
                    assignment[idx] = ci
                else:
                    # Point already assigned — pick nearest center
                    old_center = centers[assignment[idx]]
                    new_center = centers[ci]
                    d_old = (x[idx] - old_center[0]) ** 2 + (y[idx] - old_center[1]) ** 2
                    d_new = (x[idx] - new_center[0]) ** 2 + (y[idx] - new_center[1]) ** 2
                    if d_new < d_old:
                        assignment[idx] = ci

        clusters = []
        for ci in range(len(region_centers)):
            indices = np.where(assignment == ci)[0]
            if len(indices) >= min_samples:
                clusters.append(indices)

    return clusters


def _fft_find_worker(
    vx, vy, transformed_detections, radius=1 / 3600, min_obs=6, min_arc_length=1.5, min_nights=3
):
    """Ray-serializable worker that uses FFT-based density peak clustering."""
    return _cluster_velocity_find_worker(
        vx,
        vy,
        transformed_detections,
        radius=radius,
        min_obs=min_obs,
        min_arc_length=min_arc_length,
        min_nights=min_nights,
        point_cluster_fn=_find_clusters_fft,
        alg_name="FFT",
    )


_fft_find_remote = ray.remote(_fft_find_worker)
_fft_find_remote.options(num_returns=1, num_cpus=1)


class VelocityGridFFT(VelocityGridBase):
    """
    Clustering algorithm that performs a velocity-grid sweep with
    FFT-based density peak detection at each grid point.

    Rasterizes the de-rotated point cloud into a 2D histogram, convolves
    with a circular kernel, and finds peaks above the ``min_obs``
    threshold. Potentially faster than DBSCAN for large, dense fields.

    See `VelocityGridBase` for parameter documentation.
    """

    @property
    def _alg_name(self) -> str:
        return "FFT"

    def _point_cluster_fn(self) -> Callable:
        return _find_clusters_fft

    def _make_ray_remote(self) -> ray.remote_function.RemoteFunction:
        return _fft_find_remote
