import copy
import yaml
import logging

__all__ = [
    "_handleUserConfig",
    "Configuration",
    "Config"
]

logger = logging.getLogger("thor")

### DEFAULT CONFIGURATION

MIN_OBS = 5
MIN_ARC_LENGTH = 1.0
CONTAMINATION_PERCENTAGE = 20
BACKEND = "PYOORB"
BACKEND_KWARGS = {}
NUM_JOBS = "auto"
PARALLEL_BACKEND = "mp"

DEFAULT_RANGE_SHIFT_CONFIG = {
    "cell_area" : 1000,
    "num_jobs" : NUM_JOBS,
    "backend" : BACKEND,
    "backend_kwargs" : BACKEND_KWARGS,
    "parallel_backend" : PARALLEL_BACKEND
}

DEFAULT_CLUSTER_LINK_CONFIG = {
    "vx_range" : [-0.1, 0.1],
    "vy_range" : [-0.1, 0.1],
    "vx_bins" : 300,
    "vy_bins" : 300,
    "vx_values" : None,
    "vy_values" : None,
    "eps" : 5/3600,
    "min_obs" : MIN_OBS,
    "min_arc_length" : MIN_ARC_LENGTH,
    "num_jobs" : NUM_JOBS,
    "alg" : "dbscan",
    "parallel_backend" : PARALLEL_BACKEND
}

DEFAULT_IOD_CONFIG = {
    "min_obs" : MIN_OBS,
    "min_arc_length" : MIN_ARC_LENGTH,
    "contamination_percentage" : CONTAMINATION_PERCENTAGE,
    "rchi2_threshold" : 1000,
    "observation_selection_method" : "combinations",
    "iterate" : False,
    "light_time" : True,
    "linkage_id_col" : "cluster_id",
    "identify_subsets" : False,
    "chunk_size" : 10,
    "num_jobs" : NUM_JOBS,
    "backend" : BACKEND,
    "backend_kwargs" : BACKEND_KWARGS,
    "parallel_backend" : PARALLEL_BACKEND
}

DEFAULT_OD_CONFIG = {
    "min_obs" : MIN_OBS,
    "min_arc_length" : MIN_ARC_LENGTH,
    "contamination_percentage" : CONTAMINATION_PERCENTAGE,
    "rchi2_threshold" : 10,
    "delta" : 1e-6,
    "max_iter" : 5,
    "method" : "central",
    "fit_epoch" : False,
    "test_orbit" : None,
    "chunk_size" : 10,
    "num_jobs" : NUM_JOBS,
    "backend" : BACKEND,
    "backend_kwargs" : BACKEND_KWARGS,
    "parallel_backend" : PARALLEL_BACKEND
}

DEFAULT_ODP_CONFIG = {
    "min_obs" : MIN_OBS,
    "min_arc_length" : MIN_ARC_LENGTH,
    "contamination_percentage" : 0.0,
    "rchi2_threshold" : 5,
    "eps" : 1/3600,
    "delta" : 1e-8,
    "max_iter" : 5,
    "method" : "central",
    "fit_epoch" : False,
    "orbits_chunk_size" : 10,
    "observations_chunk_size" : 100000,
    "num_jobs" : NUM_JOBS,
    "backend" : BACKEND,
    "backend_kwargs" : BACKEND_KWARGS,
    "parallel_backend" : PARALLEL_BACKEND
}

def _handleUserConfig(user_dict, default_dict):
    out_dict = copy.deepcopy(user_dict)
    for key, value in default_dict.items():
        if key not in out_dict.keys():
            out_dict[key] = value
    return out_dict

class Configuration:

    def __init__(
            self,
            range_shift_config=None,
            cluster_link_config=None,
            iod_config=None,
            od_config=None,
            odp_config=None,
            min_obs=None,
            min_arc_length=None,
            contamination_percentage=None,
            backend=None,
            backend_kwargs=None,
            num_jobs=None,
            parallel_backend=None,
            ):

        if range_shift_config is None:
            self.RANGE_SHIFT_CONFIG = DEFAULT_RANGE_SHIFT_CONFIG
        else:
            self.RANGE_SHIFT_CONFIG = _handleUserConfig(
                range_shift_config,
                DEFAULT_RANGE_SHIFT_CONFIG
            )

        if cluster_link_config is None:
            self.CLUSTER_LINK_CONFIG = DEFAULT_CLUSTER_LINK_CONFIG
        else:
            self.CLUSTER_LINK_CONFIG = _handleUserConfig(
                cluster_link_config,
                DEFAULT_CLUSTER_LINK_CONFIG
            )

        if iod_config is None:
            self.IOD_CONFIG = DEFAULT_IOD_CONFIG
        else:
            self.IOD_CONFIG = _handleUserConfig(
                iod_config,
                DEFAULT_IOD_CONFIG
            )

        if od_config is None:
            self.OD_CONFIG = DEFAULT_OD_CONFIG
        else:
            self.OD_CONFIG = _handleUserConfig(
                od_config,
                DEFAULT_OD_CONFIG
            )

        if odp_config is None:
            self.ODP_CONFIG = DEFAULT_ODP_CONFIG
        else:
            self.ODP_CONFIG = _handleUserConfig(
                odp_config,
                DEFAULT_ODP_CONFIG
            )

        self._conf = {}
        if min_obs is not None:
            self.MIN_OBS = min_obs
            self._conf["MIN_OBS"] = self.MIN_OBS
            components = [
                self.CLUSTER_LINK_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("min_obs", min_obs, components)

        if min_arc_length is not None:
            self.MIN_ARC_LENGTH = min_arc_length
            self._conf["MIN_ARC_LENGTH"] = self.MIN_ARC_LENGTH
            components = [
                self.CLUSTER_LINK_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("min_arc_length", min_arc_length, components)

        if contamination_percentage is not None:
            self.CONTAMINATION_PERCENTAGE = contamination_percentage
            self._conf["CONTAMINATION_PERCENTAGE"] = self.CONTAMINATION_PERCENTAGE
            components = [
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("contamination_percentage", contamination_percentage, components)

        if backend is not None:
            self.BACKEND = backend
            self._conf["BACKEND"] = self.BACKEND
            components = [
                self.RANGE_SHIFT_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("backend", backend, components)

        if backend_kwargs is not None:
            self.BACKEND_KWARGS = backend_kwargs
            self._conf["BACKEND_KWARGS"] = self.BACKEND_KWARGS
            components = [
                self.RANGE_SHIFT_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("backend_kwargs", backend_kwargs, components)

        if num_jobs is not None:
            self.NUM_JOBS = num_jobs
            self._conf["NUM_JOBS"] = self.NUM_JOBS
            components = [
                self.RANGE_SHIFT_CONFIG,
                self.CLUSTER_LINK_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("num_jobs", num_jobs, components)

        if parallel_backend is not None:
            self.PARALLEL_BACKEND = parallel_backend
            self._conf["PARALLEL_BACKEND"] = self.PARALLEL_BACKEND
            components = [
                self.RANGE_SHIFT_CONFIG,
                self.CLUSTER_LINK_CONFIG,
                self.IOD_CONFIG,
                self.OD_CONFIG,
                self.ODP_CONFIG
            ]
            self.updateConfigs("parallel_backend", parallel_backend, components)

        self._conf["RANGE_SHIFT_CONFIG"] = self.RANGE_SHIFT_CONFIG
        self._conf["CLUSTER_LINK_CONFIG"] = self.CLUSTER_LINK_CONFIG
        self._conf["IOD_CONFIG"] = self.IOD_CONFIG
        self._conf["OD_CONFIG"] = self.OD_CONFIG
        self._conf["ODP_CONFIG"] = self.ODP_CONFIG
        return

    def __eq__(self, other):
        """
        Make sure all pipeline component dictionaries are the equal.
        """
        eq = True
        if self.RANGE_SHIFT_CONFIG != other.RANGE_SHIFT_CONFIG:
            eq = False
        if self.CLUSTER_LINK_CONFIG != other.CLUSTER_LINK_CONFIG:
            eq = False
        if self.IOD_CONFIG != other.IOD_CONFIG:
            eq = False
        if self.OD_CONFIG != other.OD_CONFIG:
            eq = False
        if self.ODP_CONFIG != other.ODP_CONFIG:
            eq = False
        return eq

    def updateConfigs(self, name, value, components):
        logger.warning(f"Setting pipeline components to use {name}: {value}.")
        for config in components:
            config[name] = value
        return

    def toYaml(self, file_name):
        """
        Save configuration to a YAML file.

        Parameters
        ----------
        file_name : str
            Path to file.
        """
        with open(file_name, "w") as config_out:
            yaml.dump(self._conf, config_out, sort_keys=False)
        return

    @classmethod
    def fromYaml(cls, file_name):
        """
        Read configuration from YAML file.

        Parameters
        ----------
        file_name : str
            Path to file.
        """
        with open(file_name, "r") as config_in:
            conf_in = yaml.load(config_in, Loader=yaml.FullLoader)

        conf = {}
        for key, values in conf_in.items():
            conf[key.lower()] = values

        config = Configuration(
            **conf
        )
        return config

Config = Configuration()