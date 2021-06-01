import numpy as np

__all__ = ["Config"]

class Config:
    """
    Config: Holds configuration settings.

    Of interest to the user are two main attributes:
        columnMapping : This dictionary should define the data
            column names of the user's data relative to the
            internally used names.
        oorbDirectory : Oorb install location should be defined
            here.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    MIN_OBS = 5
    MIN_ARC_LENGTH = 1.0
    CONTAMINATION_PERCENTAGE = 20
    BACKEND = "PYOORB"
    BACKEND_KWARGS = {}
    NUM_JOBS = "auto"
    PARALLEL_BACKEND = "mp"

    RANGE_SHIFT_CONFIG = {
        "cell_area" : 1000,
        "num_jobs" : NUM_JOBS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
        "parallel_backend" : PARALLEL_BACKEND
    }

    CLUSTER_LINK_CONFIG = {
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

    IOD_CONFIG = {
        "min_obs" : MIN_OBS,
        "min_arc_length" : MIN_ARC_LENGTH,
        "contamination_percentage" : CONTAMINATION_PERCENTAGE,
        "rchi2_threshold" : 1000,
        "observation_selection_method" : "combinations",
        "iterate" : False,
        "light_time" : True,
        "linkage_id_col" : "cluster_id",
        "identify_subsets" : True,
        "num_jobs" : NUM_JOBS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
        "parallel_backend" : PARALLEL_BACKEND
    }

    OD_CONFIG = {
        "min_obs" : MIN_OBS,
        "min_arc_length" : MIN_ARC_LENGTH,
        "contamination_percentage" : CONTAMINATION_PERCENTAGE,
        "rchi2_threshold" : 10,
        "delta" : 1e-6,
        "max_iter" : 5,
        "method" : "central",
        "fit_epoch" : False,
        "test_orbit" : None,
        "num_jobs" : NUM_JOBS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
        "parallel_backend" : PARALLEL_BACKEND
    }

    ODP_CONFIG = {
        "min_obs" : MIN_OBS,
        "min_arc_length" : MIN_ARC_LENGTH,
        "contamination_percentage" : 0.0,
        "rchi2_threshold" : 5,
        "eps" : 1/3600,
        "delta" : 1e-8,
        "max_iter" : 5,
        "method" : "central",
        "fit_epoch" : False,
        "orbits_chunk_size" : 1,
        "observations_chunk_size" : 100000,
        "num_jobs" : NUM_JOBS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
        "parallel_backend" : PARALLEL_BACKEND
    }