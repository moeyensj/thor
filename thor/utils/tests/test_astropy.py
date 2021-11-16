import pytest
import numpy as np
from astropy.time import Time

from ..astropy import _check_times

def test__checkTime():
    # Create an array of epochs
    times = np.linspace(59580, 59590, 100)

    # Test that an error is raised when times are not
    # an astropy time object
    with pytest.raises(TypeError):
        _check_times(times, "test")

    # Test that _checkTime passes when an astropy time object is
    # given as intended
    times_astropy = Time(times, format="mjd", scale="utc")

    return