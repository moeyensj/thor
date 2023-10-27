from ..config import Config


def test_Config_set_min_obs():
    config = Config()
    config.set_min_obs(3)
    assert config.cluster_min_obs == 3
    assert config.iod_min_obs == 3
    assert config.od_min_obs == 3
    assert config.arc_extension_min_obs == 3


def test_Config_set_min_arc_length():
    config = Config()
    config.set_min_arc_length(3)
    assert config.cluster_min_arc_length == 3
    assert config.iod_min_arc_length == 3
    assert config.od_min_arc_length == 3
    assert config.arc_extension_min_arc_length == 3
