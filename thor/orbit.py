import logging
import numpy as np
from numba import jit

from .projections import cartesianToGnomonic
from .coordinates import transformCoordinates
from .coordinates import _convertSphericalToCartesian

X_AXIS = np.array([1., 0., 0.])
Y_AXIS = np.array([0., 1., 0.])
Z_AXIS = np.array([0., 0., 1.])

logger = logging.getLogger(__name__)

__all__ = [
    "calcNae",
    "calcDelta",
    "calcXae",
    "calcXa",
    "calcNhat",
    "calcR1",
    "calcR2",
    "TestOrbit"
]

class TestOrbit:
    """
    TestOrbit: Class that calculates and stores the rotation matrices
    for a guess of heliocentric distance and velocity. To be used in
    tandem with the Cell class.

    Parameters
    ----------
    cartesian : `~numpy.ndarray` (1, 6)
        Cartesian ecliptic orbital elements with postions in units of au
        and velocities in units of au per day.
    t0 : `~astropy.time.core.Time` (1)
        Epoch at which orbital elements are defined.
    """
    def __init__(self, cartesian, epoch):
        self.cartesian = cartesian
        self.epoch = epoch

    def prepare(self):
        """
        Calculate rotation matrices.

        Populates the following class properties:
            n_hat : vector normal to the plane of orbit
            R1 : rotation matrix to rotate towards x-y plane
            R2 : rotation matrix to rotate towards x-axis
            M : final rotation matrix

        Returns
        -------
        None
        """
        logger.debug("Calculating vector normal to plane of orbit...")
        self.n_hat = calcNhat(self.cartesian[:3].reshape(1, -1))[0, :]

        logger.debug("Calculating R1 rotation matrix...")
        self.R1 = calcR1(self.n_hat)
        self.x_a_xy = np.array(self.R1 @ self.cartesian[:3])

        logger.debug("Calculating R2 rotation matrix...")
        self.R2 = calcR2(self.x_a_xy)

        logger.debug("Calculating final rotation matrix...")
        self.M = self.R2 @ self.R1
        return

    def applyToObservations(self, observations):
        """
        Apply the prepared rotations to the given observations. Adds the gnomonic
        plane coordinates to observations (columns: theta_x_deg, theta_y_deg)

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            DataFrame of observations defined at the same epoch as this test orbit,
            to project into the test orbit's frame.

        Returns
        -------
        None
        """
        logger.debug("Applying rotation matrices to observations...")
        logger.debug("Converting to ecliptic coordinates...")
        coords_eq = observations[["RA_deg", "Dec_deg"]].values
        coords_eq = np.hstack([np.ones((len(coords_eq), 1)), coords_eq])
        coords_ec = transformCoordinates(coords_eq,
            "equatorial",
            "ecliptic",
            representation_in="spherical",
            representation_out="spherical"
        )

        logger.debug("Calculating object to observer unit vector...")
        n_ae = calcNae(coords_ec[:, 1:3])
        x_e = observations[["obs_x", "obs_y", "obs_z"]].values

        r = np.linalg.norm(self.cartesian[:3])
        logger.debug(f"Calculating object to observer distance assuming r = {r} AU...")
        delta = calcDelta(r, x_e, n_ae)

        logger.debug("Calculating object to observer position vector...")
        x_ae = calcXae(delta, n_ae)

        logger.debug("Calculating heliocentric object position vector...")
        x_a = calcXa(x_ae, x_e)

        logger.debug("Applying rotation matrix M to heliocentric object position vector...")
        coords_cart_rotated = np.array(self.M @ x_a.T).T

        logger.debug("Performing gnomonic projection...")
        gnomonic_coords = cartesianToGnomonic(coords_cart_rotated)

        observations["obj_x'"] = x_a[:, 0]
        observations["obj_y'"] = x_a[:, 1]
        observations["obj_z'"] = x_a[:, 2]
        observations["obj_x''"] = coords_cart_rotated[:, 0]
        observations["obj_y''"] = coords_cart_rotated[:, 1]
        observations["obj_z''"] = coords_cart_rotated[:, 2]
        observations["theta_x_deg"] = np.degrees(gnomonic_coords[:, 0])
        observations["theta_y_deg"] = np.degrees(gnomonic_coords[:, 1])
        observations["test_obj_x"] = self.cartesian[0]
        observations["test_obj_y"] = self.cartesian[1]
        observations["test_obj_z"] = self.cartesian[2]
        observations["test_obj_vx"] = self.cartesian[3]
        observations["test_obj_vy"] = self.cartesian[4]
        observations["test_obj_vz"] = self.cartesian[5]

        coords_rot_test = self.M @ self.cartesian[:3].T
        coords_rot_testv = self.M @ self.cartesian[3:].T
        observations["test_obj_x''"] = coords_rot_test[0]
        observations["test_obj_y''"] = coords_rot_test[1]
        observations["test_obj_z''"] = coords_rot_test[2]
        observations["test_obj_vx''"] = coords_rot_testv[0]
        observations["test_obj_vy''"] = coords_rot_testv[1]
        observations["test_obj_vz''"] = coords_rot_testv[2]
        return

    def applyToEphemeris(self, ephemeris):
        """
        Apply the prepared rotations to the given ephemerides. Adds the gnomonic
        plane coordinates to observations (columns: theta_x_deg, theta_y_deg, vtheta_x, and vtheta_y)

        Parameters
        ----------
        ephemeris : `~pandas.DataFrame`
            DataFrame of ephemeris generated by a THOR backend defined at the same epoch as this test orbit,
            to project into the test orbit's frame.

        Returns
        -------
        None
        """
        raise NotImplementedError

@jit("f8[:,:](f8[:,:])", nopython=True, cache=True)
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
    coords_ec_ang : `~numpy.ndarray` (N, 2)
        Ecliptic longitude and latitude in degrees.

    Returns
    -------
    N_ae : `~numpy.ndarray` (N, 3)
        Cartesian unit vector in direction of provided
        angular coordinates.
    """
    N = len(coords_ec_ang)
    rho = np.ones(N)
    lon = np.radians(coords_ec_ang[:, 0])
    lat = np.radians(coords_ec_ang[:, 1])
    velocities = np.zeros(len(rho))
    x, y, z, vx, vy, vz = _convertSphericalToCartesian(rho, lon, lat, velocities, velocities, velocities)

    n_ae = np.zeros((N, 3))
    n_ae[:, 0] = x
    n_ae[:, 1] = y
    n_ae[:, 2] = z
    return n_ae

@jit("f8[:](f8, f8[:,:], f8[:,:])", nopython=True, cache=True)
def calcDelta(r, x_e, n_ae):
    """
    Calculate topocentric distance to the asteroid.

    Parameters
    ----------
    r : float (1)
        Heliocentric/barycentric distance in arbitrary units.
    x_e : `~numpy.ndarray` (N, 3)
        Topocentric position vector in same units as r.
    n_ae : `~numpy.ndarray` (N, 3)
        Unit vector in direction of asteroid from the topocentric position
        in same units as r.

    Returns
    -------
    delta : `~numpy.ndarray` (N)
        Distance from topocenter to asteroid in units of r.
    """
    N = len(x_e)
    delta = np.zeros(N)
    rsq = r**2
    for i in range(N):
        n_ae_i = np.ascontiguousarray(n_ae[i])
        x_e_i = np.ascontiguousarray(x_e[i])
        ndotxe = np.dot(n_ae_i, x_e_i)
        delta[i] = - ndotxe + np.sqrt(ndotxe**2 + rsq - np.linalg.norm(x_e_i)**2)
    return delta

@jit("f8[:,:](f8[:], f8[:,:])", nopython=True, cache=True)
def calcXae(delta, n_ae):
    """
    Calculate the topocenter to asteroid position vector.

    Parameters
    ----------
    delta : float
        Distance from the topocenter to asteroid in arbitrary units.
    n_ae : `~numpy.ndarray` (3)
        Unit vector in direction of asteroid from the topocentric position
        in same units as delta.

    Returns
    -------
    x_ae : `~numpy.ndarray` (N, 3)
        Topocenter to asteroid position vector in units of delta.
    """
    x_ae = np.zeros_like(n_ae)
    for i, (delta_i, n_ae_i) in enumerate(zip(delta, n_ae)):
        x_ae[i] = delta_i * n_ae_i
    return x_ae

@jit("f8[:,:](f8[:,:],f8[:,:])", nopython=True, cache=True)
def calcXa(x_ae, x_e):
    """
    Calculate the asteroid position vector.

    Parameters
    ----------
    x_ae : `~numpy.ndarray` (3)
        Topocenter to asteroid position vector in arbitrary units.
    x_e : `~numpy.ndarray` (3)
        Topocentric position vector in same units as x_ae.

    Returns
    -------
    x_a : `~numpy.ndarray` (3)
        Asteroid position vector in units of x_ae.
    """
    return x_ae + x_e

@jit("f8[:,:](f8[:,:])", nopython=True, cache=True)
def calcNhat(x_a):
    """
    Calulate the unit vector normal to the plane of the orbit.

    Parameters
    ----------
    x_a : `~numpy.ndarray` (N, 3)
        Asteroid position vector in arbitrary units.

    Returns
    -------
    n_hat : `~numpy.ndarray` (N, 3)
        Unit vector normal to plane of orbit.

    """
    n_hat = np.zeros_like(x_a)
    for i, x_a_i in enumerate(x_a):
        # Make n_a unit vector
        n_a = x_a_i / np.linalg.norm(x_a_i)
        # Find the normal to the plane of the orbit n
        n = np.cross(n_a, np.cross(Z_AXIS, n_a))
        # Make n a unit vector
        n_hat[i] = n / np.linalg.norm(n)
    return n_hat

@jit("f8[:,:](f8[:])", nopython=True, cache=True)
def calcR1(n_hat):
    """
    Calculate the rotation matrix that would rotate the
    position vector x_ae to the x-y plane.

    Parameters
    ----------
    n_hat : `~numpy.ndarray` (3)
        Unit vector normal to plane of orbit.

    Returns
    -------
    R1 : `~numpy.matrix` (3, 3)
        Rotation matrix.
    """
    n_hat_ = np.ascontiguousarray(n_hat)
    # Find the rotation axis v
    v = np.cross(n_hat_, Z_AXIS)
    # Calculate the cosine of the rotation angle, equivalent to the cosine of the
    # inclination
    c = np.dot(n_hat_, Z_AXIS)
    # Compute the skew-symmetric cross-product of the rotation axis vector v
    vp = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    # Calculate R1
    R1 = np.identity(3) + vp + np.linalg.matrix_power(vp, 2) * (1 / (1 + c))
    return R1

@jit("f8[:,:](f8[:])", nopython=True, cache=True)
def calcR2(x_a_xy):
    """
    Calculate the rotation matrix that would rotate a vector in
    the x-y plane to the x-axis.

    Parameters
    ----------
    x_a_xy : `~numpy.ndarray` (3)
        Barycentric asteroid position vector rotated to the x-y plane.

    Returns
    -------
    R2 : `~numpy.ndarray` (3, 3)
        Rotation ndarray.
    """
    x_a_xy = x_a_xy / np.linalg.norm(x_a_xy)
    # Assuming the vector x_a_xy has been normalized, and is in the xy plane.
    ca = x_a_xy[0]
    sa = x_a_xy[1]
    R2 = np.array([[ca, sa, 0.],
                   [-sa, ca, 0.],
                   [0., 0., 1.]])
    return R2