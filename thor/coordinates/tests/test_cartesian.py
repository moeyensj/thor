import numpy as np

from ..cartesian import CartesianCoordinates


def test_CartesianCoordinates_rotate():
    """
    Test rotation of Cartesian coordinates. Rotating coordinates and their
    covariances with an identity matrix should return the same coordinates and covariances.
    """
    N, D = 1000, 6
    values = np.random.random((N, D))
    covariances = np.random.random((N, D, D))

    coords = CartesianCoordinates(
        x=values[:, 0],
        y=values[:, 1],
        z=values[:, 2],
        vx=values[:, 3],
        vy=values[:, 4],
        vz=values[:, 5],
        covariances=covariances
    )

    rot_matrix = np.identity(D)
    values_rotated, covariances_rotated = coords._rotate(rot_matrix)

    np.testing.assert_equal(values_rotated, values)
    np.testing.assert_equal(covariances_rotated, covariances)
    return