import numpy as np
from adam_core import coordinates, orbits
from adam_core.observations import detections

from thor import orbit


def rotation_matrix(coords):
    """
    Calculates a rotation matrix
    """
    n_hat = orbit.calcNhat(coords)
    r1 = orbit.calcR1(n_hat)
    x_a_xy = np.array(r1 @ coords[:3])
    r2 = orbit.calcR2(x_a_xy)
    m = r2 @ r1
    return m


def rotate_detections(
    rot_mat: np.array, det: detections.PointSourceDetections
) -> coordinates.CartesianCoordinates:
    # Convert det.ra and det.dec to unit-sphere SphericalCoordinates
    coords = coordinates.SphericalCoordinates.from_kwargs(
        rho=np.ones(len(det)),
        lon=det.ra,
        lat=det.dec,
        origin=...,  # TODO: What should this be???
        frame="equatorial",
    )
    # Convert to ecliptic coordinates...
    coordinates.transform_coordinates(
        coords=coords,
        representation_out=coordinates.SphericalCoordinates,
        frame_out="ecliptic",
        origin_out="SUN",  # FIXME: is this right??
    )
