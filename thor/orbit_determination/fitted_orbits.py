import uuid
from typing import List, Literal, Optional, Tuple

import numpy as np
import pyarrow as pa
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

    def assign_duplicate_observations(
        self, orbit_members: "FittedOrbitMembers"
    ) -> Tuple["FittedOrbits", "FittedOrbitMembers"]:
        """
        Assigns observations that have been assigned to multiple orbits to the orbit with t
        he most observations, longest arc length, and lowest reduced chi2.

        Parameters
        ----------
        orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
            Fitted orbit members.

        Returns
        -------
        filtered : `~thor.orbit_determination.FittedOrbits`
            Fitted orbits with duplicate assignments removed.
        filtered_orbit_members : `~thor.orbit_determination.FittedOrbitMembers`
            Fitted orbit members with duplicate assignments removed.
        """
        # Sort by number of observations, arc length, and reduced chi2
        # Here we assume that orbits that generally have more observations, longer arc lengths, and lower reduced chi2 are better
        # as candidates for assigning detections that have been assigned to multiple orbits
        sorted = self.sort_by(
            [
                ("num_obs", "descending"),
                ("arc_length", "descending"),
                ("reduced_chi2", "ascending"),
            ]
        )

        # Extract the orbit IDs from the sorted table
        orbit_ids = sorted.orbit_id.unique()

        # Calculate the order in which these orbit IDs appear in the orbit_members table
        order_in_orbits = pc.index_in(orbit_members.orbit_id, orbit_ids)

        # Create an index into the orbit_members table and append the order_in_orbits column
        orbit_members_table = (
            orbit_members.flattened_table()
            .append_column("index", pa.array(np.arange(len(orbit_members))))
            .append_column("order_in_orbits", order_in_orbits)
        )

        # Drop the residual values (a list column) due to: https://github.com/apache/arrow/issues/32504
        orbit_members_table = orbit_members_table.drop_columns(["residuals.values"])

        # Sort orbit members by the orbit IDs (in the same order as the orbits table)
        orbit_members_table = orbit_members_table.sort_by(
            [("order_in_orbits", "ascending")]
        )

        # Now group by the orbit ID and aggregate the index column to get the first index for each orbit ID
        indices = (
            orbit_members_table.group_by("obs_id", use_threads=False)
            .aggregate([("index", "first")])
            .column("index_first")
        )

        # Use the indices to filter the orbit_members table and then use the resulting orbit IDs to filter the orbits table
        filtered_orbit_members = orbit_members.take(indices)
        filtered = self.apply_mask(
            pc.is_in(self.orbit_id, filtered_orbit_members.orbit_id)
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
