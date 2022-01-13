import numpy as np
from typing import (
    List,
    Optional,
    Union
)
from astropy.time import Time

from ..utils import Indexable
from ..coordinates import Coordinates

class Observers(Indexable):
    """
    Observers

    Stores observation times and coordinates for observers.

    Observers can be defined in the following ways:
    observers = Observers(["I11"], times=observation_times)
    observers = Observers(["I11", "I41"], times=observation_times)
    observers = Observers(["I11", "I41"], times=[observation_times_I11, observation_times_I41])

    """
    def __init__(self,
        codes : List,
        times : Optional[Union[Time, List]] = None,
        coordinates : Optional[Coordinates] = None
    ):
        # Case 1: codes and times are both a list -- each observatory code
        # has a list of unique observation times.
        if isinstance(codes, list) and isinstance(times, list):
            assert len(codes) == len(times)
            self._codes = np.array([c for c, t in zip(codes, times) for ti in t])
            self._times = Time(np.concatenate(times))

        # Case 2: codes is a list but times is a an astropy time object -- each observatory code
        # shares the same observation times (useful for testing).
        elif isinstance(codes, list) and isinstance(times, Time):
            self._codes = np.array([c for c in codes for i in range(len(times))])
            self._times = Time(
                np.concatenate([times for i in range(len(codes))])
            )

        # Case 3: codes is a numpy array and times is an astropy time object -- each observation time
        # has a corresponding observatory code (these codes can be duplicated)
        elif isinstance(codes, np.ndarray) and isinstance(times, Time):
            self._codes = codes
            self._times = times

        assert len(self._codes) == len(self._times)

        # Sort by observation times and then by observatory codes
        sorted_ind = np.lexsort((self._codes, self._times))
        self._times = self._times[sorted_ind]
        self._codes = self._codes[sorted_ind]
        return

    @property
    def codes(self):
        return self._codes