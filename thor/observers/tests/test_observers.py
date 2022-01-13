import numpy as np

from astropy.time import Time

from ..observers import Observers

def test_Observers_constructor():

    # Test Case 2a : list of observatory codes and single astropy time object
    observatory_codes = ["I11"]
    observation_times = Time(
        np.array([59000.0, 59001.0, 59002.0]),
        scale="utc",
        format="mjd"
    )

    observers = Observers(observatory_codes, times=observation_times)
    times_expected = Time(
        np.array([59000.0, 59001.0, 59002.0]),
        scale="utc",
        format="mjd"
    )
    codes_expected = np.array(["I11", "I11", "I11"])

    np.testing.assert_equal(observers.codes, codes_expected)

    # Test Case 2b : list of observatory codes and single astropy time object
    observatory_codes = ["I11", "I41"]
    observation_times = Time(
        np.array([59000.0, 59001.0, 59002.0]),
        scale="utc",
        format="mjd"
    )

    observers = Observers(observatory_codes, times=observation_times)
    times_expected = Time(
        np.array([59000.0, 59000.0, 59001.0, 59001.0, 59002.0, 59002.0]),
        scale="utc",
        format="mjd"
    )
    codes_expected = np.array(["I11", "I41", "I11", "I41", "I11", "I41"])
    np.testing.assert_equal(observers.codes, codes_expected)
