import pytest
import numpy as np
from astropy.time import Time

from ..observations import calcLinkingWindow, calcNight

def test_calcLinkingWindow():

    # Times are all in August 2021
    time = Time([59426 + 1/24], scale="utc", format="mjd")
    times = time + np.arange(0, 30)

    # If the linking window exceeds the range of the observations
    # then they all should be in the first linking window
    length = 30
    linking_window = calcLinkingWindow(times, length)
    np.testing.assert_equal(linking_window, np.ones(length, dtype=int))

    # If the linking window is 15 days then we expect 15 observations
    # in linking window 1 and 15 in window 2
    length = 15
    linking_window = calcLinkingWindow(times, length)
    np.testing.assert_equal(linking_window[0:15], np.ones(length, dtype=int))
    np.testing.assert_equal(linking_window[15:30], 2*np.ones(length, dtype=int))

    # If the linking window is 10 days then we expect 10 observations
    # in linking window 1 and 10 in window 2 and 10 in window 3
    length = 10
    linking_window = calcLinkingWindow(times, length)
    np.testing.assert_equal(linking_window[0:10], np.ones(length, dtype=int))
    np.testing.assert_equal(linking_window[10:20], 2*np.ones(length, dtype=int))
    np.testing.assert_equal(linking_window[20:30], 3*np.ones(length, dtype=int))

    return

def test_calcLinkingWindow_raises():

    # Times are all in August 2021
    time = Time([59426 + 1/24], scale="utc", format="mjd")
    times = time + np.arange(0, 30)

    # Passing a plain array should raise a TypeError
    with pytest.raises(TypeError):
        linking_window = calcLinkingWindow(times.mjd, 10)

    return

def test_calcNight_london():
    # Lets assume three days of observations were made
    # somewhere in London with observations occuring
    # +- 5 hours around midnight
    dts = np.linspace(-5, 5, 20)
    dts /= 24.
    offset = 0/24
    days = [59001, 59002, 59003]
    times = np.zeros(len(dts) * len(days))
    for i, day in enumerate(days):
        times[i * len(dts): (i+1) * len(dts)] = day + dts + offset
    observation_times = Time(times, scale="utc", format="mjd")

    night = calcNight(observation_times, offset)
    assert np.all(night[0:20] == 59000)
    assert np.all(night[20:40] == 59001)
    assert np.all(night[40:60] == 59002)
    return

def test_calcNight_seattle():
    # Lets assume three days of observations were made
    # somewhere in Seattle with observations occuring
    # +- 5 hours around midnight
    dts = np.linspace(-5, 5, 20)
    dts /= 24.
    offset = -7/24
    days = [59001, 59002, 59003]
    times = np.zeros(len(dts) * len(days))
    for i, day in enumerate(days):
        times[i * len(dts): (i+1) * len(dts)] = day + dts + offset
    observation_times = Time(times, scale="utc", format="mjd")

    night = calcNight(observation_times, offset)
    assert np.all(night[0:20] == 59000)
    assert np.all(night[20:40] == 59001)
    assert np.all(night[40:60] == 59002)
    return

def test_calcNight_shanghai():
    # Lets assume three days of observations were made
    # somewhere in Shanghai with observations occuring
    # +- 5 hours around midnight
    dts = np.linspace(-5, 5, 20)
    dts /= 24.
    offset = 8/24
    days = [59001, 59002, 59003]
    times = np.zeros(len(dts) * len(days))
    for i, day in enumerate(days):
        times[i * len(dts): (i+1) * len(dts)] = day + dts + offset
    observation_times = Time(times, scale="utc", format="mjd")

    night = calcNight(observation_times, offset)
    assert np.all(night[0:20] == 59000)
    assert np.all(night[20:40] == 59001)
    assert np.all(night[40:60] == 59002)
    return