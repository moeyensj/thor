from ..constants import Constants as c
from .coordinates import Coordinates
from .cartesian import CartesianCoordinates
from .spherical import SphericalCoordinates
from .keplerian import KeplerianCoordinates

TRANSFORM_EQ2EC = c.TRANSFORM_EQ2EC
TRANSFORM_EC2EQ = c.TRANSFORM_EC2EQ

__all__ = [
    "transformCoordinates",
]

def transformCoordinates(coords: Coordinates, representation_out: str, frame_out: str = "ecliptic"):
    """
    Transform coordinates between frames ('ecliptic', 'equatorial') and/or representations ('cartesian', 'spherical').
    Coordinates may include only positions or they may also include velocities.

    Parameters
    ----------
    coords : `~thor.coordinates.Coordinates`
        Coordinates to transform between representations and frames.
    representation_out : {'cartesian', 'spherical', 'keplerian'}
        Desired coordinate type or representation of the output coordinates.
    frame_out : {'equatorial', 'ecliptic'}
        Desired reference frame of the output coordinates.

    Returns
    -------
    coords_out :

    Raises
    ------
    ValueError
        If frame_in, frame_out are not one of 'equatorial', 'ecliptic'.
        If representation_in, representation_out are not one of 'cartesian', 'spherical', 'keplerian'.
    """
    # Check that coords is a thor.coordinates.Coordinates object
    if not isinstance(coords, (CartesianCoordinates, SphericalCoordinates, KeplerianCoordinates)):
        err = (
            "Coords of type {} are not supported.\n"
            "Supported coordinates are:\n"
            "  CartesianCoordinates\n"
            "  SphericalCoordinates\n"
            "  KeplerianCoordinates\n"
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

    if frame_out != "equatorial" and frame_out != "ecliptic":
        raise ValueError("".join(frame_err).format("frame_out"))

    # Check that representation_in and representation_out are one of cartesian
    # or spherical, raise errors otherwise
    representation_err = [
        "{} should be one of:\n",
        "'cartesian', 'spherical', keplerian"
    ]
    if representation_out not in ("cartesian", "spherical", "keplerian"):
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
        else:
            pass

    # At this point, some form of transformation is going to occur so
    # convert the coords to Cartesian if they aren't already
    if isinstance(coords, CartesianCoordinates):
        cartesian = coords
    else:
        cartesian = coords.to_cartesian()

    if coords.frame != frame_out:
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
    elif representation_out == "keplerian":
        coords_out = KeplerianCoordinates.from_cartesian(cartesian)
    elif representation_out == "cartesian":
        coords_out = cartesian

    return coords_out

