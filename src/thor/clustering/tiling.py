"""
Spatial tiling pre-filter for velocity-grid clustering.

After de-rotating observations by a velocity hypothesis, the spatial field
is mostly empty. This module divides the field into tiles and identifies
only those tiles with enough observations to potentially contain a cluster,
dramatically reducing the work for downstream clustering algorithms.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class TileSpec:
    """Specification for a single spatial tile."""

    tile_id: int
    # Core region (no border overlap)
    core_x_lo: float
    core_x_hi: float
    core_y_lo: float
    core_y_hi: float
    # Extended region (with border for edge clusters)
    ext_x_lo: float
    ext_x_hi: float
    ext_y_lo: float
    ext_y_hi: float
    n_obs_core: int  # observation count in core region


def compute_tile_grid(
    xx: npt.NDArray[np.float64],
    yy: npt.NDArray[np.float64],
    tile_size: float,
    border_width: float,
    min_obs: int,
) -> List[TileSpec]:
    """
    Divide the shifted coordinate space into tiles and return only those
    with at least min_obs observations in their core region.

    Parameters
    ----------
    xx, yy : ndarray
        Shifted (de-rotated) coordinates.
    tile_size : float
        Side length of each tile's core region.
    border_width : float
        Width of the border region around each tile (should be >= 2 * radius).
    min_obs : int
        Minimum observations in a tile's core to keep it.

    Returns
    -------
    tiles : list of TileSpec
        Tiles with >= min_obs observations.
    """
    if len(xx) == 0:
        return []

    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()

    # Number of tiles in each dimension
    nx = max(int(np.ceil((x_max - x_min) / tile_size)), 1)
    ny = max(int(np.ceil((y_max - y_min) / tile_size)), 1)

    # If the grid is trivially small (1 tile), skip tiling
    if nx * ny <= 1:
        return [
            TileSpec(
                tile_id=0,
                core_x_lo=x_min,
                core_x_hi=x_max,
                core_y_lo=y_min,
                core_y_hi=y_max,
                ext_x_lo=x_min,
                ext_x_hi=x_max,
                ext_y_lo=y_min,
                ext_y_hi=y_max,
                n_obs_core=len(xx),
            )
        ]

    # Bin observations into tiles using integer bin indices
    bx = np.clip(((xx - x_min) / tile_size).astype(np.int32), 0, nx - 1)
    by = np.clip(((yy - y_min) / tile_size).astype(np.int32), 0, ny - 1)

    # Count observations per tile using flat index
    flat_idx = bx * ny + by
    counts = np.bincount(flat_idx, minlength=nx * ny)

    # Find tiles with enough observations
    tiles = []
    tile_id = 0
    for flat_i in range(nx * ny):
        if counts[flat_i] < min_obs:
            continue
        ix = flat_i // ny
        iy = flat_i % ny

        core_x_lo = x_min + ix * tile_size
        core_x_hi = x_min + (ix + 1) * tile_size
        core_y_lo = y_min + iy * tile_size
        core_y_hi = y_min + (iy + 1) * tile_size

        tiles.append(
            TileSpec(
                tile_id=tile_id,
                core_x_lo=core_x_lo,
                core_x_hi=core_x_hi,
                core_y_lo=core_y_lo,
                core_y_hi=core_y_hi,
                ext_x_lo=core_x_lo - border_width,
                ext_x_hi=core_x_hi + border_width,
                ext_y_lo=core_y_lo - border_width,
                ext_y_hi=core_y_hi + border_width,
                n_obs_core=int(counts[flat_i]),
            )
        )
        tile_id += 1

    return tiles


def extract_tile_observations(
    tile: TileSpec,
    xx: npt.NDArray[np.float64],
    yy: npt.NDArray[np.float64],
) -> npt.NDArray[np.intp]:
    """
    Return indices of observations that fall within the tile's extended region.

    Parameters
    ----------
    tile : TileSpec
        The tile to extract observations for.
    xx, yy : ndarray
        Shifted coordinates.

    Returns
    -------
    indices : ndarray of int
        Global indices of observations within the extended tile region.
    """
    mask = (xx >= tile.ext_x_lo) & (xx < tile.ext_x_hi) & (yy >= tile.ext_y_lo) & (yy < tile.ext_y_hi)
    return np.where(mask)[0]


def compute_auto_tile_size(
    radius: float,
    n_obs: int,
    x_extent: float,
    y_extent: float,
    target_obs_per_tile: int = 50000,
    min_tiles_per_side: int = 1,
) -> float:
    """
    Auto-compute tile size based on observation density and radius.

    Tries to create tiles that contain roughly target_obs_per_tile observations,
    with a minimum tile size of 50 * radius to ensure clusters aren't split
    excessively.

    Parameters
    ----------
    radius : float
        Clustering radius.
    n_obs : int
        Total number of observations.
    x_extent, y_extent : float
        Spatial extent of observations.
    target_obs_per_tile : int
        Target number of observations per tile.
    min_tiles_per_side : int
        Minimum number of tiles along each axis.

    Returns
    -------
    tile_size : float
        Recommended tile size.
    """
    total_area = max(x_extent * y_extent, radius * radius)
    density = n_obs / total_area

    # Tile area needed to contain target_obs_per_tile at this density
    if density > 0:
        tile_area = target_obs_per_tile / density
        tile_size_from_density = np.sqrt(tile_area)
    else:
        tile_size_from_density = max(x_extent, y_extent)

    # Minimum tile size: 50 * radius ensures clusters aren't fragmented
    min_tile_size = 50 * radius

    tile_size = max(tile_size_from_density, min_tile_size)

    # Don't make tiles larger than the full extent
    max_extent = max(x_extent, y_extent)
    if max_extent > 0:
        tile_size = min(tile_size, max_extent)

    return tile_size
