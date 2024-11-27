import logging
import pathlib
from typing import Literal, Optional, Union

from pydantic import BaseModel

logger = logging.getLogger("thor")


class Config(BaseModel):
    max_processes: Optional[int] = None
    ray_memory_bytes: int = 0
    propagator: Literal["PYOORB"] = "PYOORB"
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
    cluster_chunk_size: int = 1000
    iod_min_obs: int = 6
    iod_min_arc_length: float = 1.0
    iod_contamination_percentage: float = 20.0
    iod_rchi2_threshold: float = 100000
    iod_observation_selection_method: Literal["combinations", "first+middle+last"] = "combinations"
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


def initialize_config(
    config: Config,
    working_dir: Optional[Union[pathlib.Path, str]],
):
    """
    Compare the given configuration to the configuration in the checkpoint directory
    and raise an exception if they do not match.

    Parameters
    ----------
    config : `~thor.config.Config`
        Configuration to compare.
    working_dir : str, optional
        Directory to compare the configuration to. If None, no comparison will be made.
    """
    if working_dir is not None:
        config_path = pathlib.Path(working_dir) / "inputs/config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            checkpoint_config = Config.parse_file(config_path)
            # Compare the configurations
            if config != checkpoint_config:
                raise ValueError(
                    f"Configuration does not match configuration in checkpoint directory: {config_path}"
                )
        else:
            logger.info(f"Configuration file does not exist: {config_path}")

        # Save the new configuration
        logger.info(f"Saving configuration to {config_path}...")
        config_path.write_text(config.json(indent=4))
