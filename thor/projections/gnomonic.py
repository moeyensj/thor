import jax.numpy as jnp
from jax import (
    config,
    jit,
    vmap
)
from typing import Tuple

config.update("jax_enable_x64", True)

__all__ = [
    "_calc_gnomonic_rotation_matrix",
    "_cartesian_to_gnomonic",
    "cartesian_to_gnomonic"
]

Z_AXIS = jnp.array([0.0, 0.0, 1.0])


@jit
def _calc_gnomonic_rotation_matrix(
        coords_cartesian: jnp.ndarray
    ) -> jnp.ndarray:
    """
    Calculate rotation matrix that aligns the position vector
    of these 6D Coordinates to the x-axis, and the velocity vector in the x-y plane.

    This is a two-fold rotation, first find the rotation matrix
    that rotates the position vector to the x-y plane. Then rotate the
    position vector from the x-y plane to the x-axis.

    TODO: If the velocity vector is zero. The velocity is assumed to be a unit vector parallel to the
    direction of the x-y plane.

    Parameters
    ----------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        Cartesian coordinates to rotate so that they lie along the x-axis and its
        velocity is in the x-y plane.

    Returns
    -------
    M : `~jax.numpy.ndarray` (6, 6)
        Gnomonic rotation matrix.
    """
    r = coords_cartesian[0:3]
    v = coords_cartesian[3:6]

    rv = jnp.cross(r, v)
    n_hat = rv / jnp.linalg.norm(rv)

    # Find the rotation axis nu
    nu = jnp.cross(n_hat, Z_AXIS)

    # Calculate the cosine of the rotation angle, equivalent to the cosine of the
    # inclination
    c = jnp.dot(n_hat, Z_AXIS)

    # Compute the skew-symmetric cross-product of the rotation axis vector v
    vp = jnp.array([
        [0, -nu[2], nu[1]],
        [nu[2], 0, -nu[0]],
        [-nu[1], nu[0], 0]
    ])
    # Calculate R1
    R1 = jnp.identity(3) + vp + jnp.linalg.matrix_power(vp, 2) * (1 / (1 + c))

    r_xy = R1 @ r
    r_xy = r_xy / jnp.linalg.norm(r_xy)

    # Calculate R2
    ca = r_xy[0]
    sa = r_xy[1]
    R2 = jnp.array([[ca, sa, 0.],
                   [-sa, ca, 0.],
                   [0., 0., 1.]])

    M = jnp.zeros((6, 6), dtype=jnp.float64)
    M = M.at[0:3, 0:3].set(R2 @ R1)
    M = M.at[3:6, 3:6].set(R2 @ R1)
    return M

@jit
def _cartesian_to_gnomonic(
        coords_cartesian: jnp.ndarray,
        center_cartesian: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Project Cartesian coordinates onto a gnomonic tangent plane centered about
    central Cartesian coordinate.

    Parameters
    ----------
    coords_cartesian : `~jax.numpy.ndarray` (6)
        Cartesian coordinates to be projected onto a tangent plane centered at
        the center Cartesian coordinate.
    center_cartesian : `~jax.numpy.ndarray` (6)
        Cartesian coordinate about which to center the tangent plane projection.

    Returns
    -------
    coords_gnomonic : `~jax.numpy.ndarray` (4)
        Gnomonic longitude, latitude and their velocities in radians and radians
        per day.
    M : `~jax.numpy.ndarray` (6, 6)
        Gnomonic rotation matrix.
    """
    M = _calc_gnomonic_rotation_matrix(center_cartesian)
    coords_cartesian_ = jnp.where(jnp.isnan(coords_cartesian), 0., coords_cartesian)
    coords_rotated = M @ coords_cartesian_

    x = coords_rotated[0]
    y = coords_rotated[1]
    z = coords_rotated[2]
    vx = coords_rotated[3]
    vy = coords_rotated[4]
    vz = coords_rotated[5]

    coords_gnomonic = jnp.zeros(4, dtype=jnp.float64)
    u = y / x
    v = z / x
    vu = (x * vy - vx * y) / x**2
    vv = (x * vz - vx * z) / x**2

    coords_gnomonic = coords_gnomonic.at[0].set(u)
    coords_gnomonic = coords_gnomonic.at[1].set(v)
    coords_gnomonic = coords_gnomonic.at[2].set(vu)
    coords_gnomonic = coords_gnomonic.at[3].set(vv)

    return coords_gnomonic, M

# Vectorization Map: _cartesian_to_gnomonic
_cartesian_to_gnomonic_vmap = vmap(
    _cartesian_to_gnomonic,
    in_axes=(0, None),
    out_axes=(0, None)
)

@jit
def cartesian_to_gnomonic(
        coords_cartesian: jnp.ndarray,
        center_cartesian: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Project Cartesian coordinates onto a gnomonic tangent plane centered about
    central Cartesian coordinate.

    Parameters
    ----------
    coords_cartesian : `~jax.numpy.ndarray` (N, 6)
        Cartesian coordinates to be projected onto a tangent plane centered at
        the center Cartesian coordinate.
    center_cartesian : `~jax.numpy.ndarray` (6)
        Cartesian coordinate about which to center the tangent plane projection.

    Returns
    -------
    coords_gnomonic : `~jax.numpy.ndarray` (N, 4)
        Gnomonic longitude, latitude and their velocities in radians and radians
        per day.
    M : `~jax.numpy.ndarray` (6, 6)
        Gnomonic rotation matrix.
    """
    coords_gnomonic, M_matrix = _cartesian_to_gnomonic_vmap(
        coords_cartesian,
        center_cartesian
    )
    return coords_gnomonic, M_matrix
