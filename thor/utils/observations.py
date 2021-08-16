import numpy as np
from astropy.time import Time

from .astropy import _checkTime

__all__ = [
    "calcLinkingWindow",
    "calcNight"
]

def calcLinkingWindow(
        times: Time,
        length: float
    ) -> np.ndarray:
    """
    Calculate the linking window for each given time. Linking windows are counted
    from the earliest time. For example, if given the following times and a defined
    window size of 5 days then the times will be assigned as follows:
    Time:                    Linking Window
        2021-08-08T12:00:00.000     1
        2021-08-09T12:00:00.000     1
        2021-08-10T12:00:00.000     1
        2021-08-11T12:00:00.000     1
        2021-08-12T12:00:00.000     1
        2021-08-13T12:00:00.000     2
        2021-08-14T12:00:00.000     2
        2021-08-15T12:00:00.000     2
        2021-08-16T12:00:00.000     2
        2021-08-17T12:00:00.000     2

    Parameters
    ----------
    times : `~astropy.time.core.Time`
        Observation times that need to be assigned to a linking window.
    length : float, int
        Length of the desired linking window in units of decimal days.

    Returns
    -------
    window : `~numpy.ndarray` (N)
        The linking window number to which each time was assigned.
    """
    _checkTime(times, "times")
    dt = times.mjd - times.mjd.min()
    window = np.floor(dt / length) + 1
    return window.astype(int)

def calcNight(
        times: Time,
        local_midnight: float
    ) -> np.ndarray:
    """
    Given a set of observation times, calculates the night
    on which those observations were made.

    Parameters
    ----------
    times : `~astropy.core.Time` (N)
        Observation times.
    local_midnight : float
        The change in time from UTC at which local midnight occurs for the observer
        or observatory that made the observations. This should be a float value in units
        of decimal days. For those locations East of the Prime Meridian these
        would be positive values (e.g., Brussels would be +1/24, Shanghai would be +8/24).
        For locations West of the Prime Meridian these values should be negative
        (e.g., New York would be -4/24, Seattle would be -7/24).

    Returns
    -------
    night : `~numpy.ndarray` (N)
        Night on which the observation was made expressed as an integer MJD.
    """
    _checkTime(times, "times")
    mjd_utc = times.utc.mjd
    local_mjd = mjd_utc - local_midnight
    # Observations that occur after midnight should be assigned the night value
    # of the observations that occurred before midnight, so shift all the observations
    # 12 hours earlier and use that as the night value
    nights = local_mjd - 12/24
    nights = nights.astype(int)
    return nights
