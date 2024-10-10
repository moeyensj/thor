import numpy as np


def calculate_max_outliers(num_obs: int, min_obs: int, contamination_percentage: float) -> int:
    """
    Calculate the maximum number of allowable outliers. Linkages may contain err
    oneuos observations that need to be removed. This function calculates the maximum number of
    observations that can be removed before the linkage no longer has the minimum number
    of observations required. The contamination percentage is the maximum percentage of observations
    that allowed to be erroneous.

    Parameters
    ----------
    num_obs : int
        Number of observations in the linkage.
    min_obs : int
        Minimum number of observations required for a valid linkage.
    contamination_percentage : float
        Maximum percentage of observations that allowed to be erroneous. Range is [0, 100].

    Returns
    -------
    outliers : int
        Maximum number of allowable outliers.
    """
    assert (
        num_obs >= min_obs
    ), "Number of observations must be greater than or equal to the minimum number of observations."
    assert (
        contamination_percentage >= 0 and contamination_percentage <= 100
    ), "Contamination percentage must be between 0 and 100."

    max_outliers = num_obs * (contamination_percentage / 100)
    outliers = np.min([max_outliers, num_obs - min_obs]).astype(int)
    return outliers
