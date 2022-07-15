import numpy as np
import pandas as pd
from typing import Optional

from ..utils.indexable import Indexable
from ..coordinates.members import CoordinateMembers


class Ephemeris(CoordinateMembers):


    def __init__(self, coordinates, orbit_ids, object_ids=None):

        CoordinateMembers.__init__(
            self,
            coordinates=coordinates,
            cartesian=True,
            spherical=True,
            keplerian=True,
            cometary=True
        )

        if object_ids is not None:
            self._object_ids = object_ids
        else:
            self._object_ids = np.array(["None" for i in range(len(coordinates))])

        self._orbit_ids = orbit_ids
        Indexable.__init__(self, index=self.orbit_ids)
        return

    def __len__(self):

        N = len(pd.unique(self._index))

        return N

    @property
    def orbit_ids(self):
        return self._orbit_ids

    @property
    def object_ids(self):
        return self._object_ids

    def to_df(self,
            time_scale: str = "utc",
            coordinate_type: Optional[str] = None,
        ) -> pd.DataFrame:
        """
        Represent ephemeris as a `~pandas.DataFrame`.

        Parameters
        ----------
        time_scale : {"tdb", "tt", "utc"}
            Desired timescale of the output MJDs.
        coordinate_type : {"cartesian", "spherical"}
            Desired output representation of the orbits.

        Returns
        -------
        df : `~pandas.DataFrame`
            Pandas DataFrame containing orbits.
        """
        df = CoordinateMembers.to_df(self,
            time_scale=time_scale,
            coordinate_type=coordinate_type
        )
        df.insert(0, "orbit_id", self.orbit_ids)
        df.insert(1, "object_id", self.object_ids)

        return df