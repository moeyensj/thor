import uuid
import numpy as np
import pandas as pd
from typing import (
    Optional,
)
from collections import OrderedDict

from ..utils.indexable import Indexable
from ..coordinates.coordinates import Coordinates
from ..coordinates.members import CoordinateMembers
from ..observers.observers import (
    OBSERVER_CARTESIAN_COLS,
    Observers
)

__all__ = ["Observations"]

class Observations(CoordinateMembers):

    def __init__(self,
        coordinates: Coordinates,
        observers: Optional[Observers] = None,
        obs_ids: Optional[np.ndarray] = None,
        orbit_ids: Optional[np.ndarray] = None,
        object_ids: Optional[np.ndarray] = None
    ):

        CoordinateMembers.__init__(
            self,
            coordinates=coordinates,
            cartesian=True,
            keplerian=False,
            cometary=False,
            spherical=True,
        )

        if not isinstance(observers, Observers):
            self._observers = Observers.from_observations_coordinates(coordinates)
        else:
            self._observers = observers

        if obs_ids is not None:
            self._obs_ids = obs_ids
        else:
            self._obs_ids = np.array([uuid.uuid4().hex for i in range(len(coordinates))])

        if object_ids is not None:
            self._object_ids = object_ids
        else:
            self._object_ids = np.array(["None" for i in range(len(coordinates))])

        if orbit_ids is not None:
            self._orbit_ids = orbit_ids
        else:
            self._orbit_ids = np.array(["None" for i in range(len(coordinates))])

        Indexable.__init__(self, index="obs_ids")
        return

    @property
    def obs_ids(self):
        return self._obs_ids

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def orbit_ids(self):
        return self._orbit_ids

    @property
    def observers(self):
        return self._observers

    def to_df(self,
            time_scale: str = "utc",
            coordinate_type: str = "spherical",
        ) -> pd.DataFrame:
        """
        Represent Observations as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {"cartesian", "spherical"}
            Desired output representation of the observations.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing observations.
        """
        df = CoordinateMembers.to_df(
            self,
            time_scale=time_scale,
            coordinate_type=coordinate_type
        )

        df.insert(0, "obs_id", self.obs_ids)
        df.insert(1, "object_id", self.object_ids)

        df_observers = self.observers.to_df(
            time_scale=time_scale,
        )
        df_observers.drop(
            columns=[f"mjd_{time_scale}"],
            inplace=True,
        )
        df = df.join(df_observers)

        return df

    @classmethod
    def from_df(
            cls,
            df,
            coord_cols: Optional[OrderedDict] = None,
            origin_col: str = "origin",
            frame_col: str = "frame",
            observer_cols: Optional[OrderedDict] = OBSERVER_CARTESIAN_COLS,
            observer_origin_col: str = "obs_origin",
            observer_frame_col: str = "obs_frame",
            obs_ids_col: str = "obs_id",
            object_ids_col: str = "object_id"
        ):

        data = {}
        if obs_ids_col in df.columns.values:
            data["obs_ids"] = df[obs_ids_col].values
        else:
            data["obs_ids"] = None

        data["coordinates"] = CoordinateMembers._dict_from_df(
            df,
            cartesian=True,
            keplerian=False,
            cometary=False,
            spherical=True,
            coord_cols=coord_cols,
            origin_col=origin_col,
            frame_col=frame_col
        )["coordinates"]

        data["observers"] = Observers.from_df(
            df,
            coord_cols=observer_cols,
            origin_col=observer_origin_col,
            frame_col=observer_frame_col,
        )

        return cls(**data)




