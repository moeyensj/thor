import uuid
from typing import Optional, TypeVar, Union

import numpy as np
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    OriginCodes,
    SphericalCoordinates,
    transform_coordinates,
)
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris, Orbits
from adam_core.propagator import PYOORB

CoordinateType = TypeVar(
    "CoordinateType",
    bound=Union[
        CartesianCoordinates,
        SphericalCoordinates,
        KeplerianCoordinates,
        CometaryCoordinates,
    ],
)


class TestOrbit:
    def __init__(
        self,
        coordinates: CoordinateType,
        orbit_id: Optional[str] = None,
        object_id: Optional[str] = None,
    ):
        """
        Create a test orbit from a set of orbital elements.

        Parameters
        ----------
        coordinates :  `~adam_core.coordinates.cartesian.CartesianCoordinates`,
                       `~adam_core.coordinates.spherical.SphericalCoordinates`,
                       `~adam_core.coordinates.keplerian.KeplerianCoordinates`,
                       `~adam_core.coordinates.cometary.CometaryCoordinates`
            The orbital elements that define this test orbit. Can be any representation but will
            be stored internally as Cartesian elements.
        orbit_id : str, optional
            Orbit ID. If not provided, a random UUID will be generated.
        object_id : str, optional
            Object ID, if it exists.
        """
        # Test orbits should be singletons
        assert len(coordinates) == 1

        # Test orbit selection will likely occur in a non-Cartesian coordinate system
        # so we should accept any coordinate system and convert to Cartesian as the
        # most stable representation
        if not isinstance(coordinates, CartesianCoordinates):
            cartesian_coordinates = coordinates.to_cartesian()
        else:
            cartesian_coordinates = coordinates

        if orbit_id is not None:
            self.orbit_id = orbit_id
        else:
            self.orbit_id = uuid.uuid4().hex

        if object_id is not None:
            self.object_id = object_id

        self._orbit = Orbits.from_kwargs(
            orbit_id=[self.orbit_id],
            object_id=[self.object_id],
            coordinates=cartesian_coordinates,
        )

    @classmethod
    def from_orbits(cls, orbits):
        assert len(orbits) == 1
        return cls(
            orbits.coordinates, orbits.orbit_id[0].as_py(), orbits.object_id[0].as_py()
        )

    @property
    def orbit(self):
        return self._orbit

    def propagate(self, times, propagator=PYOORB(), max_processes=1) -> Orbits:
        """
        Propagate this test orbit to the given times.

        Parameters
        ----------
        times : `~astropy.time.core.Time`
            Times to which to propagate the orbit.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.

        Returns
        -------
        propagated_orbit : `~adam_core.orbits.orbits.Orbits`
            The test orbit propagated to the given times.
        """
        return propagator.propagate_orbits(
            self.orbit, times, max_processes=max_processes, chunk_size=1
        )

    def generate_ephemeris(
        self, observers, propagator=PYOORB(), max_processes=1
    ) -> qv.MultiKeyLinkage[Ephemeris, Observers]:
        """
        Generate ephemeris for this test orbit at the given observers.

        Parameters
        ----------
        observers : `~adam_core.observers.Observers`
            Observers from which to generate ephemeris.
        propagator : `~adam_core.propagator.propagator.Propagator`, optional
            Propagator to use to propagate the orbit. Defaults to PYOORB.
        num_processes : int, optional
            Number of processes to use to propagate the orbit. Defaults to 1.

        Returns
        -------
        ephemeris : qv.MultiKeyLinkage[
                `~adam_core.orbits.ephemeris.Ephemeris`,
                `~adam_core.observers.observers.Observers`]
            The ephemeris of the test orbit at the given observers.
        """
        return propagator.generate_ephemeris(
            self.orbit, observers, max_processes=max_processes, chunk_size=1
        )


def assume_heliocentric_distance(
    r_mag: float, coords: SphericalCoordinates, origin_coords: CartesianCoordinates
) -> SphericalCoordinates:
    """
    Given a heliocentric distance, for all coordinates that do not have a topocentric distance defined (rho), calculate
    the topocentric distance assuming the coordinates are located at the given heliocentric distance.

    Parameters
    ----------
    r_mag : float
        Heliocentric distance to assume for the coordinates with missing topocentric distance. This is
        typically the same distance as the heliocentric distance of test orbit at the time
        of the coordinates.
    coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
        Coordinates to assume the heliocentric distance for.
    origin_coords : `~adam_core.coordinates.cartesian.CartesianCoordinates`
        Heliocentric coordinates of the origin of the topocentric coordinates.

    Returns
    -------
    coords : `~adam_core.coordinates.spherical.SphericalCoordinates`
        Coordinates with the missing topocentric distance replaced with the calculated topocentric distance.
    """
    assert len(origin_coords) == 1
    assert np.all(origin_coords.origin == OriginCodes.SUN)

    # Extract the topocentric distance and topocentric radial velocity from the coordinates
    rho = coords.rho.to_numpy(zero_copy_only=False)
    vrho = coords.vrho.to_numpy(zero_copy_only=False)

    # Transform the coordinates to the ecliptic frame by assuming they lie on a unit sphere
    # (this assumption will only be applied to coordinates with missing rho values)
    coords_eq_unit = coords.to_unit_sphere(only_missing=True)
    coords_ec = transform_coordinates(
        coords_eq_unit, SphericalCoordinates, frame_out="ecliptic"
    )

    # Transform the coordinates to cartesian and calculate the unit vectors pointing
    # from the origin to the coordinates
    coords_ec_xyz = coords_ec.to_cartesian()
    unit_vectors = coords_ec_xyz.r_hat

    # Calculate the topocentric distance such that the heliocentric distance to the coordinate
    # is r_mag
    dotprod = np.sum(unit_vectors * origin_coords.r, axis=1)
    delta = -dotprod + np.sqrt(dotprod**2 + r_mag**2 - origin_coords.r_mag**2)

    # Where rho was not defined, replace it with the calculated topocentric distance
    coords_ec = coords_ec.set_column("rho", np.where(np.isnan(rho), delta, rho))
    coords_ec = coords_ec.set_column("vrho", vrho)

    return coords_ec
