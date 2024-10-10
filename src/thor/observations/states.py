import hashlib
from typing import Union

import numpy as np
import pyarrow as pa
from adam_core.coordinates import CartesianCoordinates, SphericalCoordinates


def calculate_state_ids(coordinates: Union[SphericalCoordinates, CartesianCoordinates]) -> pa.Int64Array:
    """
    Calculate the state IDs for a set of coordinates. States are defined as unique time and origin
    combinations. This function will return a state ID for each coordinate in the input coordinates
    array. State IDs are computed in asencding order of time and origin code.

    Parameters
    ----------
    coordinates
        The coordinates for which to calculate the state IDs.

    Returns
    -------
    state_ids
        The state IDs for each coordinate in the input coordinates array.
    """
    # Extract the time and origin columns
    table = coordinates.flattened_table()

    # Append index column so we can maintain the original order
    table = table.append_column(pa.field("index", pa.int64()), pa.array(np.arange(0, len(table))))

    # Select only the relevant columns
    time_origins = table.select(["time.days", "time.nanos", "origin.code"])

    # Group the detections by the observatory code and the detection times and then grab the unique ones
    unique_time_origins = time_origins.group_by(["time.days", "time.nanos", "origin.code"]).aggregate([])

    # Now sort the unique states by the time and origin code
    unique_time_origins = unique_time_origins.sort_by(
        [
            ("time.days", "ascending"),
            ("time.nanos", "ascending"),
            ("origin.code", "ascending"),
        ]
    )

    # For each unique state assign a unique state ID
    unique_time_origins = unique_time_origins.append_column(
        pa.field("state_id", pa.int64()),
        pa.array(np.arange(0, len(unique_time_origins))),
    )

    # Drop the covariance values since ListColumns cannot be joined
    table = table.drop(["covariance.values"])

    # Join the states back to the original coordinates and sort by the original index
    coordinates_with_states = table.join(
        unique_time_origins, ["time.days", "time.nanos", "origin.code"]
    ).sort_by([("index", "ascending")])

    # Now return the state IDs
    return coordinates_with_states.column("state_id").combine_chunks()


def calculate_state_id_hash(day: int, nanos: int, observatory_code: str):
    """
    Create a hash by deterministicly combining the day, nanos, observatory code
    """
    # Use hashlib md5
    return hashlib.md5(f"{day}{nanos}{observatory_code}".encode("utf-8")).hexdigest()


def calculate_state_id_hashes(
    coordinates: Union[SphericalCoordinates, CartesianCoordinates]
) -> pa.StringArray:
    """
    Calculate the state ID hashes for a set of coordinates. State ID hashes are defined as unique
    time and origin combinations. This function will return a state ID hash for each coordinate in
    the input coordinates array. State ID hashes are computed in asencding order of time and origin
    code.

    Parameters
    ----------
    coordinates
        The coordinates for which to calculate the state ID hashes.

    Returns
    -------
    state_id_hashes
        The state ID hashes for each coordinate in the input coordinates array.
    """
    hash_inputs = coordinates.flattened_table().select(["time.days", "time.nanos", "origin.code"])
    hash_inputs = [
        hash_inputs["time.days"].to_pylist(),
        hash_inputs["time.nanos"].to_pylist(),
        hash_inputs["origin.code"].to_pylist(),
    ]

    state_id_hashes = []
    for day, nanos, observatory_code in zip(*hash_inputs):
        state_id_hashes.append(calculate_state_id_hash(day, nanos, observatory_code))

    state_id_hashes = pa.array(state_id_hashes, pa.large_string())

    return state_id_hashes
