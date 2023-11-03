import uuid
from typing import List, Literal, Optional, Tuple

import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.residuals import Residuals
from adam_core.orbits import Orbits

from ..utils.quivr import drop_duplicates

__all__ = [
    "FittedOrbits",
    "FittedOrbitMembers",
]


class FittedOrbits(qv.Table):

    orbit_id = qv.StringColumn(default=lambda: uuid.uuid4().hex)
    object_id = qv.StringColumn(nullable=True)
    coordinates = CartesianCoordinates.as_column()
    arc_length = qv.Float64Column()
    num_obs = qv.Int64Column()
    chi2 = qv.Float64Column()
    reduced_chi2 = qv.Float64Column()
    improved = qv.BooleanColumn(nullable=True)

    def to_orbits(self) -> Orbits:
        """
        Convert fitted orbits to orbits that can be used by
        a Propagator.

        Returns
        -------
        orbits : `~adam_core.orbits.Orbits`
            Orbits.
        """
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )

    def drop_duplicates(
        self,
        orbit_members: "FittedOrbitMembers",
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last"] = "first",
    ) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
        """
        Drop duplicate orbits from the fitted orbits and remove
        the corresponding orbit members.

        Parameters
        ----------
        orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
            Fitted orbit members.
        subset : list of str, optional
            Subset of columns to consider when dropping duplicates. If not specified all the columns
            specifying unique state are used: time, x, y, z, vx, vy, vz.
        keep : {'first', 'last'}, default 'first'
            If there are duplicate rows then keep the first or last row.

        Returns
        -------
        filtered : `~thor.orbit_determination.FittedOrbits`
            Fitted orbits without duplicates.
        filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
            Fitted orbit members without duplicates.
        """
        if subset is None:
            subset = [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.x",
                "coordinates.y",
                "coordinates.z",
                "coordinates.vx",
                "coordinates.vy",
                "coordinates.vz",
            ]

        filtered = drop_duplicates(self, subset=subset, keep=keep)
        filtered_orbit_members = orbit_members.apply_mask(
            pc.is_in(orbit_members.orbit_id, filtered.orbit_id)
        )
        return filtered, filtered_orbit_members


class FittedOrbitMembers(qv.Table):

    orbit_id = qv.StringColumn()
    obs_id = qv.StringColumn()
    residuals = Residuals.as_column(nullable=True)
    solution = qv.BooleanColumn(nullable=True)
    outlier = qv.BooleanColumn(nullable=True)

    def drop_outliers(self) -> "FittedOrbitMembers":
        """
        Drop outliers from the fitted orbit members.

        Returns
        -------
        fitted_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
            Fitted orbit members without outliers.
        """
        return self.apply_mask(pc.equal(self.outlier, False))
