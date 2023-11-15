import os
import pathlib
import shutil

import pytest

from ..config import Config, initialize_config


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


@pytest.fixture
def working_dir():
    path = os.path.join(os.path.dirname(__file__), "checkpoint")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


def test_compare_configs(working_dir):
    config = Config()
    config.set_min_obs(3)
    config.set_min_arc_length(3)

    # Test that if the checkpoint directory does not exist, we do not raise an exception
    initialize_config(config, working_dir=None)

    # Test that if the checkpoint directory exists and there is no existing config, we do not raise an exception
    initialize_config(config, working_dir=working_dir)

    # Test that if the checkpoint directory exists and there is a matching config, we do not raise an exception
    existing_config_path = pathlib.Path(working_dir) / "config.json"
    existing_config_path.write_text(config.json(indent=4))
    initialize_config(config, working_dir=working_dir)

    # Test that if the checkpoint directory exists and the configuration does not match, we raise an exception
    config.set_min_obs(4)
    with pytest.raises(ValueError):
        initialize_config(config, working_dir=working_dir)
