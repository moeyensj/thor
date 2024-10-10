import numpy as np
from adam_core.constants import Constants as c
from adam_core.coordinates import (
    CartesianCoordinates,
    Origin,
    SphericalCoordinates,
    transform_coordinates,
)
from adam_core.orbits import Orbits
from adam_core.time import Timestamp
from numba import jit
from numpy import roots

from .gibbs import calcGibbs
from .herrick_gibbs import calcHerrickGibbs
from .iterators import iterateStateTransition

__all__ = [
    "_calcV",
    "_calcA",
    "_calcB",
    "_calcLambdas",
    "_calcRhos",
    "approxLangrangeCoeffs",
    "calcGauss",
    "gaussIOD",
]

MU = c.MU
C = c.C


@jit("f8(f8[:], f8[:], f8[:])", nopython=True, cache=True)
def _calcV(rho1_hat, rho2_hat, rho3_hat):
    # Vector triple product that gives the area of
    # the "volume of the parallelepiped" or according to
    # to Milani et al. 2008: 3x volume of the pyramid with vertices q, r1, r2, r3.
    # Note that vector triple product rules apply here.
    return np.dot(np.cross(rho1_hat, rho2_hat), rho3_hat)


@jit("f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8, f8, f8)", nopython=True, cache=True)
def _calcA(q1, q2, q3, rho1_hat, rho3_hat, t31, t32, t21):
    # Equation 21 from Milani et al. 2008
    return np.linalg.norm(q2) ** 3 * np.dot(np.cross(rho1_hat, rho3_hat), (t32 * q1 - t31 * q2 + t21 * q3))


@jit("f8(f8[:], f8[:], f8[:], f8[:], f8, f8, f8, f8)", nopython=True, cache=True)
def _calcB(q1, q3, rho1_hat, rho3_hat, t31, t32, t21, mu=MU):
    # Equation 19 from Milani et al. 2008
    return mu / 6 * t32 * t21 * np.dot(np.cross(rho1_hat, rho3_hat), ((t31 + t32) * q1 + (t31 + t21) * q3))


@jit("UniTuple(f8, 2)(f8, f8, f8, f8, f8)", nopython=True, cache=True)
def _calcLambdas(r2_mag, t31, t32, t21, mu=MU):
    # Equations 16 and 17 from Milani et al. 2008
    lambda1 = t32 / t31 * (1 + mu / (6 * r2_mag**3) * (t31**2 - t32**2))
    lambda3 = t21 / t31 * (1 + mu / (6 * r2_mag**3) * (t31**2 - t21**2))
    return lambda1, lambda3


@jit(
    "UniTuple(f8[:], 3)(f8, f8, f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8)",
    nopython=True,
    cache=True,
)
def _calcRhos(lambda1, lambda3, q1, q2, q3, rho1_hat, rho2_hat, rho3_hat, V):
    # This can be derived by taking a series of scalar products of the coplanarity condition equation
    # with cross products of unit vectors in the direction of the observer, in particular, see Chapter 9 in
    # Milani's book on the theory of orbit determination
    numerator = -lambda1 * q1 + q2 - lambda3 * q3
    rho1_mag = np.dot(numerator, np.cross(rho2_hat, rho3_hat)) / (lambda1 * V)
    rho2_mag = np.dot(numerator, np.cross(rho1_hat, rho3_hat)) / V
    rho3_mag = np.dot(numerator, np.cross(rho1_hat, rho2_hat)) / (lambda3 * V)
    rho1 = rho1_mag * rho1_hat
    rho2 = rho2_mag * rho2_hat
    rho3 = rho3_mag * rho3_hat
    return rho1, rho2, rho3


@jit("UniTuple(f8, 2)(f8, f8, f8)", nopython=True, cache=True)
def approxLangrangeCoeffs(r_mag, dt, mu=MU):
    # Calculate the Lagrange coefficients (Gauss f and g series)
    f = 1 - (1 / 2) * mu / r_mag**3 * dt**2
    g = dt - (1 / 6) * mu / r_mag**3 * dt**2
    return f, g


def calcGauss(r1, r2, r3, t1, t2, t3):
    """
    Calculates the velocity vector at the location of the second position vector (r2) with Gauss's
    original method.

    .. math::
        f_1 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_1 - t_2)^2

        f_3 \approx 1 - \frac{1}{2}\frac{\mu}{r_2^3} (t_3 - t_2)^2

        g_1 \approx (t_1 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_1 - t_2)^2

        g_3 \approx (t_3 - t_2) - \frac{1}{6}\frac{\mu}{r_2^3} (t_3 - t_2)^2

        \vec{v}_2 = \frac{1}{f_1 g_3 - f_3 g_1} (-f_3 \vec{r}_1 + f_1 \vec{r}_3)


    For more details on theory see Chapter 5 in Howard D. Curtis' "Orbital Mechanics for
    Engineering Students".

    Parameters
    ----------
    r1 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 1 in cartesian coordinates in units
        of AU.
    r2 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 2 in cartesian coordinates in units
        of AU.
    r3 : `~numpy.ndarray` (3)
        Heliocentric position vector at time 3 in cartesian coordinates in units
        of AU.
    t1 : float
        Time at r1. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.
    t2 : float
        Time at r2. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.
    t3 : float
        Time at r3. Units of MJD or JD work or any decimal time format (one day is 1.00) as
        long as all times are given in the same format.

    Returns
    -------
    v2 : `~numpy.ndarray` (3)
        Velocity of object at position r2 at time t2 in units of AU per day.
    """
    t12 = t1 - t2
    t32 = t3 - t2
    r2_mag = np.linalg.norm(r2)
    f1, g1 = approxLangrangeCoeffs(r2_mag, t12)
    f3, g3 = approxLangrangeCoeffs(r2_mag, t32)
    v2 = (1 / (f1 * g3 - f3 * g1)) * (-f3 * r1 + f1 * r3)
    return v2


def gaussIOD(
    coords,
    observation_times,
    coords_obs,
    velocity_method="gibbs",
    light_time=True,
    iterate=True,
    iterator="state transition",
    mu=MU,
    max_iter=10,
    tol=1e-15,
):
    """
    Compute up to three intial orbits using three observations in angular equatorial
    coordinates.

    Parameters
    ----------
    coords : `~numpy.ndarray` (3, 2)
        RA and Dec of three observations in units of degrees.
    observation_times : `~numpy.ndarray` (3)
        Times of the three observations in units of decimal days (MJD or JD for example).
    coords_obs : `~numpy.ndarray` (3, 2)
        Heliocentric position vector of the observer at times t in units of AU.
    velocity_method : {'gauss', gibbs', 'herrick+gibbs'}, optional
        Which method to use for calculating the velocity at the second observation.
        [Default = 'gibbs']
    light_time : bool, optional
        Correct for light travel time.
        [Default = True]
    iterate : bool, optional
        Iterate initial orbit using universal anomaly to better approximate the
        Lagrange coefficients.
    mu : float, optional
        Gravitational parameter (GM) of the attracting body in units of
        AU**3 / d**2.
    max_iter : int, optional
        Maximum number of iterations over which to converge to solution.
    tol : float, optional
        Numerical tolerance to which to compute chi using the Newtown-Raphson
        method.

    Returns
    -------
    epochs : `~numpy.ndarry` (<3)
        Epochs in units of decimal days at which preliminary orbits are
        defined.
    orbits : `~numpy.ndarray` ((<3, 6) or (0))
        Up to three preliminary orbits (as cartesian state vectors).
    """
    coords = transform_coordinates(
        SphericalCoordinates.from_kwargs(
            lon=coords[:, 0],
            lat=coords[:, 1],
            origin=Origin.from_kwargs(code=np.full(len(coords), "Unknown", dtype="object")),
            frame="equatorial",
        ).to_unit_sphere(),
        representation_out=CartesianCoordinates,
        frame_out="ecliptic",
    )
    rho = coords.values[:, :3]

    rho1_hat = rho[0, :]
    rho2_hat = rho[1, :]
    rho3_hat = rho[2, :]
    q1 = coords_obs[0, :]
    q2 = coords_obs[1, :]
    q3 = coords_obs[2, :]
    q2_mag = np.linalg.norm(q2)

    # Make sure rhohats are unit vectors
    rho1_hat = rho1_hat / np.linalg.norm(rho1_hat)
    rho2_hat = rho2_hat / np.linalg.norm(rho2_hat)
    rho3_hat = rho3_hat / np.linalg.norm(rho3_hat)

    t1 = observation_times[0]
    t2 = observation_times[1]
    t3 = observation_times[2]
    t31 = t3 - t1
    t21 = t2 - t1
    t32 = t3 - t2

    A = _calcA(q1, q2, q3, rho1_hat, rho3_hat, t31, t32, t21)
    B = _calcB(q1, q3, rho1_hat, rho3_hat, t31, t32, t21, mu=mu)
    V = _calcV(rho1_hat, rho2_hat, rho3_hat)
    coseps2 = np.dot(q2, rho2_hat) / q2_mag
    C0 = V * t31 * q2_mag**4 / B
    h0 = -A / B

    if np.isnan(C0) or np.isnan(h0):
        return np.array([])

    # Find roots to eighth order polynomial
    all_roots = roots(
        [
            C0**2,
            0,
            -(q2_mag**2) * (h0**2 + 2 * C0 * h0 * coseps2 + C0**2),
            0,
            0,
            2 * q2_mag**5 * (h0 + C0 * coseps2),
            0,
            0,
            -(q2_mag**8),
        ]
    )

    # Keep only positive real roots (which should at most be 3)
    r2_mags = np.real(all_roots[np.isreal(all_roots) & (np.real(all_roots) >= 0)])
    num_solutions = len(r2_mags)
    if num_solutions == 0:
        return np.array([])

    orbits = []
    epochs = []
    for r2_mag in r2_mags:
        lambda1, lambda3 = _calcLambdas(r2_mag, t31, t32, t21, mu=mu)
        rho1, rho2, rho3 = _calcRhos(lambda1, lambda3, q1, q2, q3, rho1_hat, rho2_hat, rho3_hat, V)

        if np.dot(rho2, rho2_hat) < 0:
            continue

        # Test if we get the same rho2 as using equation 22 in Milani et al. 2008
        rho2_mag = (h0 - q2_mag**3 / r2_mag**3) * q2_mag / C0
        # np.testing.assert_almost_equal(np.dot(rho2_mag, rho2_hat), rho2)

        r1 = q1 + rho1
        r2 = q2 + rho2
        r3 = q3 + rho3

        if velocity_method == "gauss":
            v2 = calcGauss(r1, r2, r3, t1, t2, t3)
        elif velocity_method == "gibbs":
            v2 = calcGibbs(r1, r2, r3)
        elif velocity_method == "herrick+gibbs":
            v2 = calcHerrickGibbs(r1, r2, r3, t1, t2, t3)
        else:
            raise ValueError("velocity_method should be one of {'gauss', 'gibbs', 'herrick+gibbs'}")

        epoch = t2
        orbit = np.concatenate([r2, v2])

        if iterate is True:
            if iterator == "state transition":
                orbit = iterateStateTransition(
                    orbit,
                    t21,
                    t32,
                    q1,
                    q2,
                    q3,
                    rho1,
                    rho2,
                    rho3,
                    light_time=light_time,
                    mu=mu,
                    max_iter=max_iter,
                    tol=tol,
                )

        if light_time is True:
            rho2_mag = np.linalg.norm(orbit[:3] - q2)
            lt = rho2_mag / C
            epoch -= lt

        if np.linalg.norm(orbit[3:]) >= C:
            continue

        if (np.linalg.norm(orbit[:3]) > 300.0) or (np.linalg.norm(orbit[3:]) > 1.0):
            # Orbits that crash PYOORB:
            # 58366.84446725786 : 9.5544354809296721e+01  1.4093228616761269e+01 -6.6700146960148423e+00 -6.2618123281073522e+01 -9.4167879481188717e+00  4.4421501034359023e+0
            continue

        orbits.append(orbit)
        epochs.append(epoch)

    epochs = np.array(epochs)
    orbits = np.array(orbits)
    if len(orbits) > 0:
        epochs = epochs[~np.isnan(orbits).any(axis=1)]
        orbits = orbits[~np.isnan(orbits).any(axis=1)]

        return Orbits.from_kwargs(
            coordinates=CartesianCoordinates.from_kwargs(
                x=orbits[:, 0],
                y=orbits[:, 1],
                z=orbits[:, 2],
                vx=orbits[:, 3],
                vy=orbits[:, 4],
                vz=orbits[:, 5],
                time=Timestamp.from_mjd(epochs, scale="utc"),
                origin=Origin.from_kwargs(code=np.full(len(orbits), "SUN", dtype="object")),
                frame="ecliptic",
            )
        )

    else:
        return Orbits.empty()
