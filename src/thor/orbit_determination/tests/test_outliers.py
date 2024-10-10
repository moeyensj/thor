import pytest

from ..outliers import calculate_max_outliers


def test_calculate_max_outliers():
    # Test that the function returns the correct number of outliers given
    # the number of observations, minimum number of observations, and
    # contamination percentage in a few different cases.
    min_obs = 3
    num_obs = 10
    contamination_percentage = 50
    outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)
    assert outliers == 5

    min_obs = 6
    num_obs = 10
    contamination_percentage = 50
    outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)
    assert outliers == 4

    min_obs = 6
    num_obs = 6
    contamination_percentage = 50
    outliers = calculate_max_outliers(num_obs, min_obs, contamination_percentage)
    assert outliers == 0


def test_calculate_max_outliers_assertions():
    # Test that the function raises an assertion error when the number of observations
    # is less than the minimum number of observations.
    min_obs = 10
    num_obs = 6
    contamination_percentage = 50
    with pytest.raises(
        AssertionError,
        match=r"Number of observations must be greater than or equal to the minimum number of observations.",
    ):
        calculate_max_outliers(num_obs, min_obs, contamination_percentage)

    # Test that the function raises an assertion error when the contamination percentage
    # is less than 0.
    min_obs = 10
    num_obs = 10
    contamination_percentage = -50
    with pytest.raises(AssertionError, match=r"Contamination percentage must be between 0 and 1."):
        calculate_max_outliers(num_obs, min_obs, contamination_percentage)
