from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Config:
    max_processes: Optional[int] = None
    propagator: str = "PYOORB"
    parallel_backend: Literal["cf"] = "cf"
    cell_radius: float = 10
    vx_min: float = -0.1
    vx_max: float = 0.1
    vy_min: float = -0.1
    vy_max: float = 0.1
    vx_bins: int = 300
    vy_bins: int = 300
    cluster_radius: float = 0.005
    cluster_min_obs: int = 6
    cluster_min_arc_length: float = 1.0
    cluster_algorithm: Literal["hotspot_2d", "dbscan"] = "dbscan"
    iod_min_obs: int = 6
    iod_min_arc_length: float = 1.0
    iod_contamination_percentage: float = 20.0
    iod_rchi2_threshold: float = 100000
    iod_observation_selection_method: Literal[
        "combinations", "first+middle+last"
    ] = "combinations"
    iod_chunk_size: int = 10
    od_min_obs: int = 6
    od_min_arc_length: float = 1.0
    od_contamination_percentage: float = 20.0
    od_rchi2_threshold: float = 10
    od_delta: float = 1e-6
    od_max_iter: int = 10
    od_chunk_size: int = 10
    arc_extension_min_obs: int = 6
    arc_extension_min_arc_length: float = 1.0
    arc_extension_contamination_percentage: float = 0.0
    arc_extension_rchi2_threshold: float = 10
    arc_extension_radius: float = 1 / 3600
    arc_extension_chunk_size: int = 100

    def set_min_obs(self, min_obs: int):
        """
        Set the minimum number of observations for all stages of the pipeline.

        Parameters
        ----------
        min_obs
            The minimum number of observations.
        """
        self.cluster_min_obs = min_obs
        self.iod_min_obs = min_obs
        self.od_min_obs = min_obs
        self.arc_extension_min_obs = min_obs

    def set_min_arc_length(self, min_arc_length: int):
        """
        Set the minimum arc length for all stages of the pipeline.

        Parameters
        ----------
        min_arc_length
            The minimum arc length.
        """
        self.cluster_min_arc_length = min_arc_length
        self.iod_min_arc_length = min_arc_length
        self.od_min_arc_length = min_arc_length
        self.arc_extension_min_arc_length = min_arc_length
