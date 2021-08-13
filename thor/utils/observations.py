import numpy as np
from .astropy import _checkTime

__all__ = ["calcLinkingWindow"]

def calcLinkingWindow(times, length):
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