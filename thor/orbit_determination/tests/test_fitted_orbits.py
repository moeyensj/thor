import uuid

import numpy as np
import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates

from thor.orbit_determination.fitted_orbits import FittedOrbitMembers, FittedOrbits


@pytest.fixture
def simple_orbits():
    # Creating a simple FittedOrbits instance with non-nullable fields
    num_entries = 3
    return FittedOrbits.from_kwargs(
        orbit_id=["1", "2", "3"],
        object_id=[uuid.uuid4().hex for _ in range(num_entries)],  # assuming object_id can be any string
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
            frame="unspecified"
        ),
        arc_length=np.random.rand(num_entries) * 100,
        num_obs=np.random.randint(1, 50, num_entries),
        chi2=np.random.rand(num_entries),
        reduced_chi2=np.random.rand(num_entries),
        improved=pa.repeat(False, num_entries)
    )

@pytest.fixture
def simple_orbit_members():
    # Creating a simple FittedOrbitMembers instance
    num_entries = 5
    return FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "2", "3", "1", "2"],
        obs_id=["1", "2", "3", "4", "5"],
        residuals=[None] * num_entries,  # Assuming 'residuals' can be nullable
        solution=[None] * num_entries,  # Assuming 'solution' can be nullable
        outlier=[None] * num_entries  # Assuming 'outlier' can be nullable
    )

@pytest.fixture
def duplicate_orbit_members():
    # Creating FittedOrbitMembers instance with duplicate observations
    return FittedOrbitMembers.from_kwargs(
        orbit_id=["1", "2", "1", "2", "1", "3"],
        obs_id=["1", "1", "2", "2", "3", "3"],  # Duplicates in 'obs_id'
        residuals=[None] * 6,  # Assuming 'residuals' can be nullable
        solution=[None] * 6,  # Assuming 'solution' can be nullable
        outlier=[None] * 6  # Assuming 'outlier' can be nullable
    )

def test_with_duplicate_observations(simple_orbits, duplicate_orbit_members):
    # Test handling of duplicate observations
    filtered_orbits, filtered_members = simple_orbits.assign_duplicate_observations(duplicate_orbit_members)

    # Check if the duplicates have been handled correctly
    # Assuming that duplicates are removed based on the sorting criteria of 'num_obs', 'arc_length', 'reduced_chi2'
    unique_obs_ids = set(filtered_members.obs_id)
    assert len(unique_obs_ids) == len(filtered_members)  # Ensure no duplicates in 'obs_id'


def test_with_no_duplicates(simple_orbits, simple_orbit_members):
    # Test with no duplicates in the data
    filtered_orbits, filtered_members = simple_orbits.assign_duplicate_observations(simple_orbit_members)
    assert len(filtered_orbits) == len(simple_orbits)
    assert len(filtered_members) == len(simple_orbit_members)

def test_empty_data():
    # Test with empty FittedOrbits and FittedOrbitMembers
    empty_orbits = FittedOrbits.empty()
    empty_members = FittedOrbitMembers.empty()
    filtered_orbits, filtered_members = empty_orbits.assign_duplicate_observations(empty_members)
    assert len(filtered_orbits) == 0
    assert len(filtered_members) == 0
