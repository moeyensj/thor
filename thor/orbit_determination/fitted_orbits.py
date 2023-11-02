import uuid

import quivr as qv
from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.residuals import Residuals
from adam_core.orbits import Orbits

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


class FittedOrbitMembers(qv.Table):

    orbit_id = qv.StringColumn()
    obs_id = qv.StringColumn()
    residuals = Residuals.as_column(nullable=True)
    solution = qv.BooleanColumn()
    outlier = qv.BooleanColumn()
