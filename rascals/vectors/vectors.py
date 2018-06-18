import numpy as np
from numpy.linalg import norm
from ..coordinates import _angularToCartesian

__all__ = ["calcNae",
           "calcDelta",
           "calcXae",
           "calcXa",
           "calcNhat",
           "calcR1",
           "calcR2"]

x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])


def calcNae(coords_ec_ang):
    """
    Convert angular ecliptic coordinates to
    to a cartesian unit vector.

    Input coordinate array should have shape (N, 2):
    np.array([[lon1, lat1],
              [lon2, lat2],
              [lon3, lat3],
                  .....
              [lonN, latN]])

    Parameters
    ----------
    coords_ec_ang : `~np.ndarray` (N, 2)
        Ecliptic longitude and latitude in radians.

    Returns
    -------
    N_ae : `~np.ndarray` (N, 3)
        Cartesian unit vector in direction of provided
        angular coordinates.
    """
    return _angularToCartesian(coords_ec_ang[:, 0], coords_ec_ang[:, 1], 1)


def calcDelta(r, x_e, n_ae):
    """
    Calculate topocentric distance to the asteroid.

    Parameters
    ----------
    r : float
        Heliocentric distance in arbitrary units.
    x_e : `~np.ndarray` (3)
        Topocentric position vector in same units as r.
    n_ae : `~np.ndarray` (3)
        Unit vector in direction of asteroid from the topocentric position
        in same units as r.

    Returns
    -------
    delta : float
        Distance from topocenter to asteroid in units of r.
    """
    return - np.dot(n_ae, x_e) + np.sqrt(norm(np.dot(n_ae, x_e))
                                         ** 2 + r**2 - norm(x_e)**2)


def calcXae(delta, n_ae):
    """
    Calculate the topocenter to asteroid position vector.

    Parameters
    ----------
    delta : float
        Distance from the topocenter to asteroid in arbitrary units.
    n_ae : `~np.ndarray` (3)
        Unit vector in direction of asteroid from the topocentric position
        in same units as delta.

    Returns
    -------
    x_ae : `~np.ndarray` (3)
        Topocenter to asteroid position vector in units of delta.
    """
    return np.dot(delta, n_ae)


def calcXa(x_ae, x_e):
    """
    Calculate the barycentric asteroid position vector.

    Parameters
    ----------
    x_ae : `~np.ndarray` (3)
        Topocenter to asteroid position vector in arbitrary units.
    x_e : `~np.ndarray` (3)
        Topocentric position vector in same units as x_ae.

    Returns
    -------
    x_a : `~np.ndarray` (3)
        Barycentric asteroid position vector in units of x_ae.
    """
    return x_ae + x_e

def calcNhat(x_a):
    """
    Calulate the unit vector normal to the plane of the orbit.
    
    Parameters
    ----------
    x_a : `~np.ndarray` (3)
        Barycentric asteroid position vector in arbitrary units.
        
    Returns
    -------
    n_hat : `~np.ndarray` (3)
        Unit vector normal to plane of orbit.
    
    """
    # Make n a unit vectorn_hat = n / norm(n)
    n_a = x_a / norm(x_a)
    # Find the normal to the plane of the orbit n
    n = np.cross(n_a, np.cross(z_axis, n_a))
    # Make n a unit vector
    n_hat = n / norm(n)
    return n_hat


def calcR1(x_a, n_hat):
    """
    Calculate the rotation matrix that would rotate the barycentric
    position vector x_ae to the x-y plane.

    Parameters
    ----------
    x_a : `~np.ndarray` (3)
        Barycentric asteroid position vector in arbitrary units.
    n_hat : `~np.ndarray` (3)
        Unit vector normal to plane of orbit.

    Returns
    -------
    R1 : `~np.matrix` (3, 3)
        Rotation matrix.
    """
    # Find the rotation axis v
    v = np.cross(n_hat, z_axis)
    # Calculate the cosine of the rotation angle, equivalent to the cosine of the
    # inclination
    c = np.dot(n_hat, z_axis)
    # Compute the skew-symmetric cross-product of the rotation axis vector v
    vp = np.matrix([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    # Calculate R1 and return
    return np.identity(3) + vp + vp**2 * (1 / (1 + c))


def calcR2(x_a_xy):
    """
    Calculate the rotation matrix that would rotate a vector in
    the x-y plane to the x-axis.

    Parameters
    ----------
    x_a_xy : `~np.ndarray` (3)
        Barycentric asteroid position vector rotated to the x-y plane.

    Returns
    -------
    R2 : `~np.matrix` (3, 3)
        Rotation matrix.
    """
    x_a_xy = x_a_xy / norm(x_a_xy)
    # Assuming the vector x_a_xy has been normalized, and is in the xy plane.
    ca = x_a_xy[0]
    sa = x_a_xy[1]
    return np.matrix([[ca, sa, 0],
                      [-sa, ca, 0],
                      [0, 0, 1]])
