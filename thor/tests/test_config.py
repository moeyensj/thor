from ..config import Configuration, _handleUserConfig


def test__handleUserConfig():

    # Define some arbitrary defaultr dictionary
    default_config = {
        "a": 1,
        "b": 2,
        "c": 3,
    }

    user_config = {"a": 4}

    config = _handleUserConfig(user_config, default_config)

    assert config["a"] == 4
    assert config["b"] == 2
    assert config["c"] == 3
    return


def test_configuration_rangeAndShift():
    Config = Configuration()

    # Configuration should be a dictionary
    assert isinstance(Config.RANGE_SHIFT_CONFIG, dict)
    kwargs = ["cell_area", "num_jobs", "backend", "backend_kwargs", "parallel_backend"]

    # Make sure all rangeAndShift kwargs are included in the dictionary
    for kwarg in kwargs:
        assert kwarg in Config.RANGE_SHIFT_CONFIG.keys()
    assert len(Config.RANGE_SHIFT_CONFIG.keys()) == len(kwargs)

    return


def test_configuration_clusterAndLink():
    Config = Configuration()

    # Configuration should be a dictionary
    assert isinstance(Config.CLUSTER_LINK_CONFIG, dict)

    # Make sure all clusterAndLink kwargs are included in the dictionary
    kwargs = [
        "vx_range",
        "vy_range",
        "vx_bins",
        "vy_bins",
        "vx_values",
        "vy_values",
        "eps",
        "min_obs",
        "min_arc_length",
        "alg",
        "num_jobs",
        "parallel_backend",
    ]
    for kwarg in kwargs:
        assert kwarg in Config.CLUSTER_LINK_CONFIG.keys()
    assert len(Config.CLUSTER_LINK_CONFIG.keys()) == len(kwargs)

    return


def test_configuration_iod():
    Config = Configuration()

    # Configuration should be a dictionary
    assert isinstance(Config.IOD_CONFIG, dict)

    # Make sure all initialOrbitDetermination kwargs are included in the dictionary
    kwargs = [
        "min_obs",
        "min_arc_length",
        "contamination_percentage",
        "rchi2_threshold",
        "observation_selection_method",
        "iterate",
        "light_time",
        "linkage_id_col",
        "identify_subsets",
        "backend",
        "backend_kwargs",
        "chunk_size",
        "num_jobs",
        "parallel_backend",
    ]
    for kwarg in kwargs:
        assert kwarg in Config.IOD_CONFIG.keys()
    assert len(Config.IOD_CONFIG.keys()) == len(kwargs)

    return


def test_configuration_od():
    Config = Configuration()

    # Configuration should be a dictionary
    assert isinstance(Config.OD_CONFIG, dict)

    # Make sure all differentialCorrection kwargs are included in the dictionary
    kwargs = [
        "min_obs",
        "min_arc_length",
        "contamination_percentage",
        "rchi2_threshold",
        "delta",
        "max_iter",
        "method",
        "fit_epoch",
        "test_orbit",
        "backend",
        "backend_kwargs",
        "chunk_size",
        "num_jobs",
        "parallel_backend",
    ]
    for kwarg in kwargs:
        assert kwarg in Config.OD_CONFIG.keys()
    assert len(Config.OD_CONFIG.keys()) == len(kwargs)

    return


def test_configuration_odp():
    Config = Configuration()

    # Configuration should be a dictionary
    assert isinstance(Config.ODP_CONFIG, dict)

    # Make sure all mergeAndExtendOrbits kwargs are included in the dictionary
    kwargs = [
        "min_obs",
        "min_arc_length",
        "contamination_percentage",
        "rchi2_threshold",
        "eps",
        "delta",
        "max_iter",
        "method",
        "fit_epoch",
        "backend",
        "backend_kwargs",
        "orbits_chunk_size",
        "observations_chunk_size",
        "num_jobs",
        "parallel_backend",
    ]
    for kwarg in kwargs:
        assert kwarg in Config.ODP_CONFIG.keys()
    assert len(Config.ODP_CONFIG.keys()) == len(kwargs)

    return


def test_configuration_min_obs_override():
    val = 1234
    Config = Configuration(min_obs=val)

    assert "min_obs" not in Config.RANGE_SHIFT_CONFIG.keys()
    assert Config.CLUSTER_LINK_CONFIG["min_obs"] == val
    assert Config.IOD_CONFIG["min_obs"] == val
    assert Config.OD_CONFIG["min_obs"] == val
    assert Config.ODP_CONFIG["min_obs"] == val
    assert Config.MIN_OBS == val

    return


def test_configuration_min_arc_length_override():
    val = 999.999
    Config = Configuration(min_arc_length=val)

    assert "min_arc_length" not in Config.RANGE_SHIFT_CONFIG.keys()
    assert Config.CLUSTER_LINK_CONFIG["min_arc_length"] == val
    assert Config.IOD_CONFIG["min_arc_length"] == val
    assert Config.OD_CONFIG["min_arc_length"] == val
    assert Config.ODP_CONFIG["min_arc_length"] == val
    assert Config.MIN_ARC_LENGTH == val

    return


def test_configuration_contamination_percentage_override():
    val = 100.0
    Config = Configuration(contamination_percentage=val)

    assert "contamination_percentage" not in Config.RANGE_SHIFT_CONFIG.keys()
    assert "contamination_percentage" not in Config.CLUSTER_LINK_CONFIG.keys()
    assert Config.IOD_CONFIG["contamination_percentage"] == val
    assert Config.OD_CONFIG["contamination_percentage"] == val
    assert Config.ODP_CONFIG["contamination_percentage"] == val
    assert Config.CONTAMINATION_PERCENTAGE == val

    return


def test_configuration_backend_override():
    val = "propagator"
    Config = Configuration(backend=val)

    assert Config.RANGE_SHIFT_CONFIG["backend"] == val
    assert "backend" not in Config.CLUSTER_LINK_CONFIG.keys()
    assert Config.IOD_CONFIG["backend"] == val
    assert Config.OD_CONFIG["backend"] == val
    assert Config.ODP_CONFIG["backend"] == val
    assert Config.BACKEND == val

    return


def test_configuration_backend_kwargs_override():
    val = {"dynamical_model": "test"}
    Config = Configuration(backend_kwargs=val)

    assert Config.RANGE_SHIFT_CONFIG["backend_kwargs"] == val
    assert "backend_kwargs" not in Config.CLUSTER_LINK_CONFIG.keys()
    assert Config.IOD_CONFIG["backend_kwargs"] == val
    assert Config.OD_CONFIG["backend_kwargs"] == val
    assert Config.ODP_CONFIG["backend_kwargs"] == val
    assert Config.BACKEND_KWARGS == val

    return


def test_configuration_num_jobs_override():
    val = 123
    Config = Configuration(num_jobs=val)

    assert Config.RANGE_SHIFT_CONFIG["num_jobs"] == val
    assert Config.CLUSTER_LINK_CONFIG["num_jobs"] == val
    assert Config.IOD_CONFIG["num_jobs"] == val
    assert Config.OD_CONFIG["num_jobs"] == val
    assert Config.ODP_CONFIG["num_jobs"] == val
    assert Config.NUM_JOBS == val

    return


def test_configuration_parallel_backend_override():
    val = "parallelizer"
    Config = Configuration(parallel_backend=val)

    assert Config.RANGE_SHIFT_CONFIG["parallel_backend"] == val
    assert Config.CLUSTER_LINK_CONFIG["parallel_backend"] == val
    assert Config.IOD_CONFIG["parallel_backend"] == val
    assert Config.OD_CONFIG["parallel_backend"] == val
    assert Config.ODP_CONFIG["parallel_backend"] == val
    assert Config.PARALLEL_BACKEND == val

    return
