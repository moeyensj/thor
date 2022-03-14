import numpy as np
import pandas as pd
from typing import (
    List,
    Optional,
    Union
)
from astropy.time import Time

from ..utils import Indexable
from ..coordinates import CartesianCoordinates
from .state import get_observer_state

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
        cartesian : Optional[CartesianCoordinates] = None
    ):
        self._cartesian = None
        if isinstance(cartesian, CartesianCoordinates) and times is not None:
            if cartesian.times is not None:
                try:
                    np.testing.assert_equal(cartesian.times.tdb.mjd, times.tdb.mjd)
                except AssertionError as e:
                    err = (
                        "CartesianCoordinates times do not match the given times."
                    )
                    raise ValueError(err)
            else:
                if isinstance(times, Time):
                    assert len(times) == len(cartesian)
                    cartesian._times = times

            self._cartesian = cartesian

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
        if self._cartesian is not None:
            assert len(self._codes) == len(self._cartesian)

        # Sort by observation times and then by observatory codes
        sorted_ind = np.lexsort((self._codes, self._times))
        self._times = self._times[sorted_ind]
        self._codes = self._codes[sorted_ind]
        if self._cartesian is not None:
            self._cartesian = self._cartesian[sorted_ind]

        return

    def iterate_unique(self):
        """
        Yield unique observatory codes and their coresponding
        observation times.

        Yield
        -----
        code : str
            MPC observatory code
        times : `~astropy.time.core.Time`
            Observation times for the specific observatory.
        """
        unique_code = np.unique(self.codes)
        for code in unique_code:
            times = self._times[np.where(self.codes == code)]
            yield code, times

    def __len__(self):
        return len(self.codes)

    @property
    def codes(self):
        return self._codes

    @property
    def times(self):
        return self._times

    @property
    def cartesian(self):

        if self._cartesian is None:

            dfs = []
            for code, times in self.iterate_unique():
                df_i = get_observer_state(
                    [code],
                    times,
                    origin="heliocenter",
                    frame="ecliptic"
                )
                dfs.append(df_i)

            cartesian_df = pd.concat(dfs, ignore_index=True)
            cartesian_df.sort_values(
                by=["mjd_utc", "observatory_code"],
                inplace=True,
                ignore_index=True
            )

            cartesian = CartesianCoordinates.from_df(
                cartesian_df,
                coord_cols={
                    "x" : "obs_x",
                    "y" : "obs_y",
                    "z" : "obs_z",
                    "vx" : "obs_vx",
                    "vy" : "obs_vy",
                    "vz" : "obs_vz",
                },
                covariance_col=None
            )
            self._cartesian = cartesian

        return self._cartesian

    def to_df(self, time_scale="utc"):

        df = self.cartesian.to_df(time_scale=time_scale)
        obs_cols = {}
        df.insert(1, "observatory_code", self.codes)
        df.rename(columns=obs_cols, inplace=True)
        return df

