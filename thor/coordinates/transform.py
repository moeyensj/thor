import logging
import numpy as np
from copy import deepcopy
from typing import Optional

from ..constants import Constants as c
from .coordinates import (
    COORD_FILL_VALUE,
    Coordinates
)
from .conversions import convert_coordinates
from .cartesian import (
    CartesianCoordinates,
    CARTESIAN_UNITS
)
from .spherical import (
    SphericalCoordinates,
    SPHERICAL_UNITS
)
from .keplerian import (
    KeplerianCoordinates,
    KEPLERIAN_UNITS
)
from .cometary import (
    CometaryCoordinates,
    COMETARY_UNITS
)

TRANSFORM_EQ2EC = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = c.TRANSFORM_EC2EQ

logger = logging.getLogger(__name__)

__all__ = [
    "transform_coordinates",
]

def transform_coordinates(
        coords: Coordinates,
        representation_out: str,
        frame_out: Optional[str] = None,
        unit_sphere: bool = True,
    ) -> Coordinates:
    """
    Transform coordinates between frames ('ecliptic', 'equatorial') and/or representations ('cartesian', 'spherical', 'keplerian').

    Parameters
    ----------
    coords : `~thor.coordinates.Coordinates`
        Coordinates to transform between representations and frames.
    representation_out : {'cartesian', 'spherical', 'keplerian', 'cometary'}
        Desired coordinate type or representation of the output coordinates.
    frame_out : {'equatorial', 'ecliptic'}
        Desired reference frame of the output coordinates.
    unit_sphere : bool
        Assume the coordinates lie on a unit sphere. In many cases, spherical coordinates may not have a value
        for radial distance or radial velocity but transforms to other representations or frames are still meaningful.
        If this parameter is set to true, then if radial distance is not defined and/or radial velocity is not defined
        then they are assumed to be 1.0 au and 0.0 au/d, respectively.

    Returns
    -------
    coords_out : `~thor.coordinates.Coordinates`
        Coordinates in desired output representation and frame.

    Raises
    ------
    ValueError
        If frame_in, frame_out are not one of 'equatorial', 'ecliptic'.
        If representation_in, representation_out are not one of 'cartesian',
            'spherical', 'keplerian', 'cometary'.
    """
    # Check that coords is a thor.coordinates.Coordinates object
    if not isinstance(coords, (
            CartesianCoordinates,
            SphericalCoordinates,
            KeplerianCoordinates,
            CometaryCoordinates)
        ):
        err = (
            "Coords of type {} are not supported.\n"
            "Supported coordinates are:\n"
            "  CartesianCoordinates\n"
            "  SphericalCoordinates\n"
            "  KeplerianCoordinates\n"
            "  CometaryCoordinates\n"
        )
        raise TypeError(err)

    # Check that frame_in and frame_out are one of equatorial
    # or ecliptic, raise errors otherwise
    frame_err = [
        "{} should be one of:\n",
        "'equatorial' or 'ecliptic'"
    ]
    if coords.frame != "equatorial" and coords.frame != "ecliptic":
        raise ValueError("".join(frame_err).format("frame_in"))

    if frame_out is not None:
        if frame_out != "equatorial" and frame_out != "ecliptic":
            raise ValueError("".join(frame_err).format("frame_out"))
    else:
        frame_out = coords.frame

    # Check that representation_in and representation_out are one of cartesian
    # or spherical, raise errors otherwise
    representation_err = [
        "{} should be one of:\n",
        "'cartesian', 'spherical', 'keplerian', 'cometary'"
    ]
    if representation_out not in ("cartesian", "spherical", "keplerian", "cometary"):
        raise ValueError("".join(representation_err).format("representation_out"))

    # If coords are already in the desired frame and representation
    # then return them unaltered
    if coords.frame == frame_out:
        if isinstance(coords, CartesianCoordinates) and representation_out == "cartesian":
            return coords
        elif isinstance(coords, SphericalCoordinates) and representation_out == "spherical":
            return coords
        elif isinstance(coords, KeplerianCoordinates) and representation_out == "keplerian":
            return coords
        elif isinstance(coords, CometaryCoordinates) and representation_out == "cometary":
            return coords
        else:
            pass

    # At this point, some form of transformation is going to occur so make a
    # copy of the coords and then convert the coords to Cartesian if they aren't already and make sure
    # the units match the default units assumed for each class
    coords_ = deepcopy(coords)

    set_rho_nan = False
    set_vrho_nan = False
    if isinstance(coords_, CartesianCoordinates):
        if not coords_.has_units(CARTESIAN_UNITS):
            logger.info("Cartesian coordinates do not have default units, converting units before transforming.")
            coords_ = convert_coordinates(coords_, CARTESIAN_UNITS)
        cartesian = coords_

    elif isinstance(coords_, SphericalCoordinates):
        if not coords_.has_units(SPHERICAL_UNITS):
            logger.info("Spherical coordinates do not have default units, converting units before transforming.")
            coords_ = convert_coordinates(coords_, SPHERICAL_UNITS)

        if representation_out == "spherical" or representation_out == "cartesian":
            if unit_sphere:
                set_rho_nan = True
                if np.all(np.isnan(coords_.rho.filled())):
                    logger.debug("Spherical coordinates have no defined radial distance (rho), assuming spherical coordinates lie on unit sphere.")
                    coords_.values[:, 0] = 1.0
                    coords._values.mask[:, 0] = 0

                set_vrho_nan = True
                if np.all(np.isnan(coords_.vrho.filled())):
                    logger.debug("Spherical coordinates have no defined radial velocity (vrho), assuming spherical coordinates lie on unit sphere with zero velocity.")
                    coords_.covariances[:, 3, :] = 0.0
                    coords_.covariances[:, :, 3] = 0.0

        cartesian = coords_.to_cartesian()

    elif isinstance(coords_, KeplerianCoordinates):
        if not coords_.has_units(KEPLERIAN_UNITS):
            logger.info("Keplerian coordinates do not have default units, converting units before transforming.")
            coords_ = convert_coordinates(coords_, KEPLERIAN_UNITS)

        cartesian = coords_.to_cartesian()

    elif isinstance(coords_, CometaryCoordinates):
        if not coords_.has_units(COMETARY_UNITS):
            logger.info("Cometary coordinates do not have default units, converting units before transforming.")
            coords_ = convert_coordinates(coords_, COMETARY_UNITS)

        cartesian = coords_.to_cartesian()

    # TODO : Add call to cartesian translate to shit the origin of the coordinates

    if coords_.frame != frame_out:
        if frame_out == "ecliptic":
            cartesian = cartesian.to_ecliptic()
        elif frame_out == "equatorial":
            cartesian = cartesian.to_equatorial()
        else:
            err = (
                "frame should be one of {'ecliptic', 'equatorial'}"
            )
            raise ValueError(err)

    if representation_out == "spherical":
        coords_out = SphericalCoordinates.from_cartesian(cartesian)

        # If we assumed the coordinates lie on a unit sphere and the
        # rho and vrho values were assumed then make sure the output coordinates
        # and covariances are set back to NaN values and masked
        if set_rho_nan:
            coords_out._values[:, 0] = COORD_FILL_VALUE
            coords_out._values[:, 0].mask = 1

        if set_vrho_nan:
            coords_out._values[:, 3] = COORD_FILL_VALUE
            coords_out._values[:, 3].mask = 1

    elif representation_out == "keplerian":
        coords_out = KeplerianCoordinates.from_cartesian(cartesian)
    elif representation_out == "cometary":
        coords_out = CometaryCoordinates.from_cartesian(cartesian)
    elif representation_out == "cartesian":
        coords_out = cartesian

    return coords_out

