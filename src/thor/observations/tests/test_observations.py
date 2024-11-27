import pyarrow as pa
import pytest

from ...config import Config
from ..observations import (
    InputObservations,
    Observations,
    _input_observations_iterator,
    convert_input_observations_to_observations,
    input_observations_to_observations_worker,
)
from ..states import calculate_state_id_hashes


@pytest.fixture
def input_observations_fixture(fixed_detections):
    return InputObservations.from_kwargs(
        id=fixed_detections.id,
        exposure_id=fixed_detections.exposure_id,
        time=fixed_detections.time,
        ra=fixed_detections.ra,
        dec=fixed_detections.dec,
        mag=fixed_detections.mag,
        filter=pa.repeat("r", len(fixed_detections.id)),
        observatory_code=pa.repeat("I11", len(fixed_detections.id)),
    )


@pytest.fixture
def input_observations_file(tmp_path, input_observations_fixture):
    path = tmp_path / "input_observations.parquet"
    input_observations_fixture.to_parquet(path)
    return str(path)


@pytest.fixture
def observations_config():
    return Config(max_processes=1)


def test_input_observations_iterator_quivr(input_observations_fixture):
    iterator = _input_observations_iterator(input_observations_fixture)
    for i in iterator:
        assert isinstance(i, InputObservations)


def test_input_observations_iterator_file(input_observations_file):
    iterator = _input_observations_iterator(input_observations_file)
    for i in iterator:
        assert isinstance(i, InputObservations)


def test_input_observations_to_observations(input_observations_fixture):
    observations = input_observations_to_observations_worker(input_observations_fixture)
    assert isinstance(observations, Observations)
    assert len(observations) == len(input_observations_fixture)


def test_convert_observations_table(input_observations_fixture, observations_config):
    observations = convert_input_observations_to_observations(input_observations_fixture, observations_config)
    assert isinstance(observations, Observations)
    assert len(observations) == len(input_observations_fixture)


def test_convert_observations_file(tmp_path, observations_config, input_observations_file):
    output = str(tmp_path / "output.parquet")
    observations = convert_input_observations_to_observations(
        input_observations_file, observations_config, output_path=output
    )
    assert isinstance(observations, str)
    assert output == str(tmp_path / "output.parquet")

    observations_from_file = Observations.from_parquet(output)
    inputs_from_file = InputObservations.from_parquet(input_observations_file)
    assert len(observations_from_file) == len(inputs_from_file)


def test_calculate_state_id_hashes(fixed_observations):
    hashes = calculate_state_id_hashes(fixed_observations.coordinates)
    assert isinstance(hashes, pa.LargeStringArray)
