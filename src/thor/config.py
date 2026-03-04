import json
import logging
import pathlib
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Union

logger = logging.getLogger("thor")


@dataclass(eq=True)
class Config:
    max_processes: Optional[int] = None
    ray_memory_bytes: int = 0
    propagator_namespace: str = "adam_assist.ASSISTPropagator"
    filter_cell_radius: Optional[float] = None
    filter_mahalanobis_distance: Optional[float] = 5.0
    filter_chunk_size: int = 100000
    cluster_vx_min: float = -0.1
    cluster_vx_max: float = 0.1
    cluster_vy_min: float = -0.1
    cluster_vy_max: float = 0.1
    cluster_vx_bins: int = 300
    cluster_vy_bins: int = 300
    cluster_radius: float = 0.005
    cluster_mahalanobis_distance: float = 3.0
    cluster_velocity_bin_separation: float = 2.0
    cluster_whiten: bool = False
    cluster_radius_multiplier: float = 5.0
    cluster_density_multiplier: float = 2.5
    cluster_min_radius: float = 1 / 3600  # 1 arcsec in degrees
    cluster_max_radius: float = 0.05  # 180 arcsec in degrees
    cluster_min_obs: int = 6
    cluster_min_arc_length: float = 1.0
    cluster_min_nights: int = 3
    cluster_rchi2_threshold: float = 1e4
    cluster_algorithm: str = "dbscan"
    cluster_chunk_size: int = 1000
    split_threshold: Optional[int] = None
    split_max_depth: int = 2
    split_method: Literal["healpixel", "eigenvalue"] = "eigenvalue"
    stop_after_stage: Optional[
        Literal[
            "filter_observations",
            "generate_ephemeris",
            "range_and_transform",
            "cluster_and_link",
            "fit_clusters",
            "initial_orbit_determination",
            "differential_correction",
            "recover_orbits",
        ]
    ] = None
    iod_min_obs: int = 6
    iod_min_arc_length: float = 1.0
    iod_contamination_percentage: float = 20.0
    iod_rchi2_threshold: float = 1e3
    iod_observation_selection_method: Literal["combinations", "first+middle+last"] = "combinations"
    iod_chunk_size: int = 10
    od_min_obs: int = 6
    od_min_arc_length: float = 1.0
    od_contamination_percentage: float = 20.0
    od_rchi2_threshold: float = 1e2
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

    def json(self, indent: Optional[int] = None) -> str:
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def parse_file(cls, path: Union[str, pathlib.Path]) -> "Config":
        path = pathlib.Path(path)
        data = json.loads(path.read_text())
        return cls(**data)


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
