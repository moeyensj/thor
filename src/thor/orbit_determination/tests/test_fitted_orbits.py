import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from adam_core.coordinates import CartesianCoordinates

from thor.orbit_determination.fitted_orbits import (
    FittedOrbitMembers,
    FittedOrbits,
    assign_duplicate_observations,
)


@pytest.fixture
def simple_orbits():
    # Creating a simple FittedOrbits instance with non-nullable fields
    num_entries = 5
    return FittedOrbits.from_kwargs(
        orbit_id=["1", "2", "3", "4", "5"],
        object_id=[uuid.uuid4().hex for _ in range(5)],
        coordinates=CartesianCoordinates.from_kwargs(
            x=np.random.rand(num_entries),
            y=np.random.rand(num_entries),
            z=np.random.rand(num_entries),
            vx=np.random.rand(num_entries),
            vy=np.random.rand(num_entries),
            vz=np.random.rand(num_entries),
            time=[None] * num_entries,  # Assuming 'time' can be nullable
            covariance=[None] * num_entries,  # Assuming 'covariance' can be nullable
            origin=[None] * num_entries,  # Assuming 'origin' can be nullable
            frame="unspecified",
        ),
        arc_length=[100, 150, 200, 250, 300],  # Specific values
        num_obs=[10, 20, 15, 25, 5],  # Specific values
        chi2=np.random.rand(5),
        reduced_chi2=[0.5, 0.4, 0.3, 0.2, 0.1],  # Specific values
        iterations=[100, 200, 300, 400, 500],  # Specific values
        success=[True, True, True, True, True],  # Specific values
        status_code=[1, 1, 1, 1, 1],  # Specific values
    )


@pytest.fixture
def no_duplicate_orbit_members():
    # Creating a simple FittedOrbitMembers instance
    num_entries = 5
    return FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "2", "3", "4", "5"],
        obs_id=["1", "2", "3", "4", "5"],
        residuals=[None] * num_entries,
        solution=[None] * num_entries,
        outlier=[None] * num_entries,
    )


@pytest.fixture
def all_duplicates_orbit_members():
    # Every observation is a duplicate
    return FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "2", "1", "2", "1", "2"],
        obs_id=["1", "1", "2", "2", "3", "3"],
        residuals=[None] * 6,
        solution=[None] * 6,
        outlier=[None] * 6,
    )


@pytest.fixture
def mixed_duplicates_orbit_members():
    # Mix of unique and duplicate observations
    return FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "2", "2", "3", "4", "5", "1"],
        obs_id=["1", "2", "3", "3", "4", "4", "5"],
        residuals=[None] * 7,
        solution=[None] * 7,
        outlier=[None] * 7,
    )


def test_all_duplicates(simple_orbits, all_duplicates_orbit_members):
    # Test when all observations are duplicates
    filtered_orbits, filtered_members = assign_duplicate_observations(
        simple_orbits, all_duplicates_orbit_members
    )
    unique_obs_ids = set(filtered_members.obs_id)
    assert len(unique_obs_ids) == len(filtered_members)  # Ensure no duplicates in 'obs_id'


def test_mixed_duplicates(simple_orbits, mixed_duplicates_orbit_members):
    # Test a mix of unique and duplicate observations
    filtered_orbits, filtered_members = assign_duplicate_observations(
        simple_orbits, mixed_duplicates_orbit_members
    )
    unique_obs_ids = set(filtered_members.obs_id)
    assert len(unique_obs_ids) == len(filtered_members)  # Ensure no duplicates in 'obs_id'

    # 1 -> 1
    # 2 -> 2
    # 3 -> 2
    # 4 -> 4
    # 5 -> 1
    # Three unique orbits: 1, 2, 4
    assert len(filtered_orbits) == 3

    # Now we assert that the correct orbits were selected based on sort order
    # Assert that filtered_members where obs_id is 2 that the orbit_id is 2
    mask = pc.equal(filtered_members.obs_id, pa.scalar("2"))
    assert pc.all(pc.equal(filtered_members.apply_mask(mask).orbit_id, pa.scalar("2"))).as_py()
    # Assert that members where obs_id is 3 that the orbit_id is also 2
    mask = pc.equal(filtered_members.obs_id, pa.scalar("3"))
    assert pc.all(pc.equal(filtered_members.apply_mask(mask).orbit_id, pa.scalar("2"))).as_py()
    # Assert that members where obs_id is 4 that the orbit_id is 4
    mask = pc.equal(filtered_members.obs_id, pa.scalar("4"))
    assert pc.all(pc.equal(filtered_members.apply_mask(mask).orbit_id, pa.scalar("4"))).as_py()


def test_with_no_duplicates(simple_orbits, no_duplicate_orbit_members):
    # Test with no duplicates in the data
    filtered_orbits, filtered_members = assign_duplicate_observations(
        simple_orbits, no_duplicate_orbit_members
    )
    assert len(filtered_orbits) == len(simple_orbits)
    assert len(filtered_members) == len(no_duplicate_orbit_members)


def test_empty_data():
    # Test with empty FittedOrbits and FittedOrbitMembers
    empty_orbits = FittedOrbits.empty()
    empty_members = FittedOrbitMembers.empty()
    filtered_orbits, filtered_members = assign_duplicate_observations(empty_orbits, empty_members)
    assert len(filtered_orbits) == 0
    assert len(filtered_members) == 0
