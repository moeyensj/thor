import numpy as np
from astropy import units as u

from .orbit_testing import __statsToErrorMessage
from .orbit_testing import _evaluateDifference

__all__ = [
    "testCartesianEpehemeris",
    "testSphericalEpehemeris",
    "testEphemeris"
]

def testCartesianEpehemeris(
        ephemeris_actual,
        ephemeris_desired,
        position_tol=1*u.m,
        velocity_tol=(1*u.mm/u.s),
        magnitude=True,
        raise_error=True
    ):
    """
    Tests that the two sets of cartesian ephemeris are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|.

    Parameters
    ----------
    ephemeris_actual : `~numpy.ndarray` (N, 3) or (N, 6)
        Array of ephemeris to compare to the desired ephemeris, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day
    ephemeris_desired : `~numpy.ndarray` (N, 3) or (N, 6)
        Array of desired ephemeris to which to compare the actual ephemeris to, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (x, y, z, r).
    velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance velocity need to satisfy. (vx, vy, vz, v).
    magnitude : bool
        Test the magnitude of the position difference
        and velocity difference vectors as opposed to testing per individual coordinate.

    Raises
    ------
    AssertionError:
        If |ephemeris_actual - ephemeris_desired| > tolerance.
    ValueError:
        If ephemeris shapes are not equal.
    ValueError:
        If coordinate dimensions are not one of 3 or 6.

    Returns
    -------
    None
    """
    any_error = False
    error_message = "\n"
    differences = {}
    statistics = {}

    if ephemeris_actual.shape != ephemeris_desired.shape:
        err = (
            "The shapes of the actual and desired ephemeris should be the same."
        )
        raise ValueError(err)

    N, D = ephemeris_actual.shape
    if D not in (3, 6):
        err = (
            "The number of coordinate dimensions should be one of 3 or 6.\n"
            "If 3 then the expected inputs are x, y, z positions in AU.\n"
            "If 6 then the expected inputs are x, y, z postions in AU\n"
            "and vx, vy, vz velocities in AU per day."
        )
        raise ValueError(err)

    # Test positions
    if magnitude:
        names = ["r"]
    else:
        names = ["x", "y", "z"]
    diff, stats, error = _evaluateDifference(
        ephemeris_actual[:, :3],
        ephemeris_desired[:, :3],
        u.AU,
        position_tol,
        magnitude=magnitude
    )
    for i, n in enumerate(names):
        differences[n] = diff[:, i]
        statistics[n] = {k : v[i] for k, v in stats.items()}

    # If any of the differences between desired and actual are
    # greater than the allowed tolerance set any_error to True
    # and build the error message
    if error:
        any_error = True
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, position_tol)
        error_message = __statsToErrorMessage(
            stats,
            error_message
        )

    if D == 6:
        # Test velocities
        if magnitude:
            names = ["v"]
        else:
            names = ["vx", "vy", "vz"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual[:, 3:],
            ephemeris_desired[:, 3:],
            (u.AU / u.d),
            velocity_tol,
            magnitude=magnitude
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, velocity_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

    if any_error and raise_error:
        raise AssertionError(error_message)

    return differences, statistics, error

def testSphericalEpehemeris(
        ephemeris_actual,
        ephemeris_desired,
        position_tol=1*u.m,
        velocity_tol=(1*u.mm/u.s),
        angle_tol=(1*u.arcsec),
        angular_velocity_tol=(1*u.arcsec / u.s),
        magnitude=True,
        raise_error=True
    ):
    """
    Tests that the two sets of cartesian ephemeris are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|.

    Parameters
    ----------
    ephemeris_actual : `~numpy.ndarray` (N, 2), (N, 3) or (N, 6)
        Array of ephemeris to compare to the desired ephemeris, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            angular velocities : degrees per day
    ephemeris_desired : `~numpy.ndarray` (N, 2), (N, 3) or (N, 6)
        Array of desired ephemeris to which to compare the actual ephemeris to, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            angular velocities : degrees per day
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (rho).
    velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance velocity need to satisfy. (vrho).
    angle_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angles need to satisfy (lon, lat).
    angular_velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angular velocities need to satisfy. (vlon, vrho).
    magnitude : bool
        Test the magnitudes of the angular difference
        and angular velocity difference vectors as opposed to testing per individual coordinate.
        If magnitude is set to True then the longitudinal coordinates (lon, vlon) are linearized by the cosine
        of the latitudinal coordinates (cos(lat)).

    Raises
    ------
    AssertionError:
        If |ephemeris_actual - ephemeris_desired| > tolerance.
    ValueError:
        If ephemeris shapes are not equal.
    ValueError:
        If coordinate dimensions are not one of 2, 3, or 6.

    Returns
    -------
    None
    """
    any_error = False
    error_message = "\n"
    differences = {}
    statistics = {}

    if ephemeris_actual.shape != ephemeris_desired.shape:
        err = (
            "The shapes of the actual and desired ephemeris should be the same."
        )
        raise ValueError(err)

    N, D = ephemeris_actual.shape
    if D not in (2, 3, 6):
        err = (
            "The number of coordinate dimensions should be one of 3 or 6.\n"
            "If 2 then the expected inputs are longitude, latitude angles in degrees.\n"
            "If 3 then the expected inputs are rho in AU, longitude and latitude\n"
            "in degrees.\n"
            "If 6 then the expected inputs are rho in AU, longitude and latitude\n"
            "in degrees, vhro in AU per day, vlon and vlat in degrees per day.\n"
        )
        raise ValueError(err)

    if D == 2:

        if magnitude:
            # linearize spherical angles
            # longitude_linear = cos(latitude) * longitude
            ephemeris_actual_ = ephemeris_actual.copy()
            ephemeris_actual_[:, 0] = np.cos(np.radians(ephemeris_actual_[:, 1])) * ephemeris_actual_[:, 0]

            ephemeris_desired_ = ephemeris_desired.copy()
            ephemeris_desired_[:, 0] = np.cos(np.radians(ephemeris_desired_[:, 1])) * ephemeris_desired_[:, 0]

        else:
            ephemeris_actual_ = ephemeris_actual.copy()
            ephemeris_desired_ = ephemeris_desired.copy()

        # Test angles
        if magnitude:
            names = ["theta"]
        else:
            names = ["lon", "lat"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_,
            ephemeris_desired_,
            u.degree,
            angle_tol,
            magnitude=magnitude
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, angle_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

    if D == 3:

        if magnitude:
            # linearize spherical angles
            # longitude_linear = cos(latitude) * longitude
            ephemeris_actual_ = ephemeris_actual.copy()
            ephemeris_actual_[:, 1] = np.cos(np.radians(ephemeris_actual_[:, 2])) * ephemeris_actual_[:, 1]

            ephemeris_desired_ = ephemeris_desired.copy()
            ephemeris_desired_[:, 1] = np.cos(np.radians(ephemeris_desired_[:, 2])) * ephemeris_desired_[:, 1]

        else:
            ephemeris_actual_ = ephemeris_actual
            ephemeris_desired_ = ephemeris_desired


        # Test positions
        names = ["rho"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, :1],
            ephemeris_desired_[:, :1],
            u.AU,
            position_tol,
            magnitude=False
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, position_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

        # Test angles
        if magnitude:
            names = ["theta"]
        else:
            names = ["lon", "lat"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, 1:3],
            ephemeris_desired_[:, 1:3],
            u.degree,
            angle_tol,
            magnitude=magnitude
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, angle_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

    if D == 6:

        if magnitude:
            # linearize spherical angles
            # longitude_linear = cos(latitude) * longitude
            # vlongitude_linear = cos(latitude) * vlongitude
            ephemeris_actual_ = ephemeris_actual.copy()
            ephemeris_actual_[:, 1] = np.cos(np.radians(ephemeris_actual_[:, 2])) * ephemeris_actual_[:, 1]
            ephemeris_actual_[:, 4] = np.cos(np.radians(ephemeris_actual_[:, 2])) * ephemeris_actual_[:, 4]

            ephemeris_desired_ = ephemeris_desired.copy()
            ephemeris_desired_[:, 1] = np.cos(np.radians(ephemeris_desired_[:, 2])) * ephemeris_desired_[:, 1]
            ephemeris_desired_[:, 4] = np.cos(np.radians(ephemeris_desired_[:, 2])) * ephemeris_desired_[:, 4]

        else:
            ephemeris_actual_ = ephemeris_actual
            ephemeris_desired_ = ephemeris_desired


        # Test positions
        names = ["rho"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, :1],
            ephemeris_desired_[:, :1],
            u.AU,
            position_tol,
            magnitude=False
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, position_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

        # Test angles
        if magnitude:
            names = ["theta"]
        else:
            names = ["lon", "lat"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, 1:3],
            ephemeris_desired_[:, 1:3],
            u.degree,
            angle_tol,
            magnitude=magnitude
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, angle_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

        # Test velocity
        names = ["vrho"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, 3:4],
            ephemeris_desired_[:, 3:4],
            u.AU / u.d,
            velocity_tol,
            magnitude=False)
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, velocity_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

        # Test angular velocities
        if magnitude:
            names = ["n"]
        else:
            names = ["vlon", "vlat"]
        diff, stats, error = _evaluateDifference(
            ephemeris_actual_[:, 3:4],
            ephemeris_desired_[:, 3:4],
            u.degree / u.d,
            angular_velocity_tol,
            magnitude=False
        )
        for i, n in enumerate(names):
            differences[n] = diff[:, i]
            statistics[n] = {k : v[i] for k, v in stats.items()}

        # If any of the differences between desired and actual are
        # greater than the allowed tolerance set any_error to True
        # and build the error message
        if error:
            any_error = True
            error_message += "{} difference (|actual - desired|) is not within {}.\n".format(names, angular_velocity_tol)
            error_message = __statsToErrorMessage(
                stats,
                error_message
            )

    if any_error and raise_error:
        raise AssertionError(error_message)

    return differences, statistics, error

def testEphemeris(
        ephemeris_actual,
        ephemeris_desired,
        ephemeris_type="spherical",
        position_tol=1*u.m,
        velocity_tol=(1*u.mm/u.s),
        angle_tol=(1*u.arcsec),
        angular_velocity_tol=(1*u.arcsec / u.s),
        magnitude=True,
        raise_error=True
    ):
    """
    Tests that the two sets of cartesian ephemeris are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|.

    Parameters
    ----------
    ephemeris_actual : `~numpy.ndarray` (N, 2), (N, 3) or (N, 6)
        Array of ephemeris to compare to the desired ephemeris, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            angular velocities : degrees per day
    ephemeris_desired : `~numpy.ndarray` (N, 2), (N, 3) or (N, 6)
        Array of desired ephemeris to which to compare the actual ephemeris to, may optionally
        include velocities.
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            angular velocities : degrees per day
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (x, y, z, rho).
    velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance velocity need to satisfy. (vx, vy, vz, vrho).
    angle_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angles need to satisfy (lon, lat).
    angular_velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angular velocities need to satisfy. (vlon, vrho).
    magnitude : bool
        Test the magnitudes of the angular difference and angular velocity difference vectors for spherical ephemeris or
        the magnitudes of the position difference and velocity difference for cartesian ephemeris.
        If magnitude is set to True and the input ephemeris type is spherical then the longitudinal coordinates (lon, vlon)
        are linearized by the cosine of the latitudinal coordinates (cos(lat)).

    Raises
    ------
    AssertionError:
        If |ephemeris_actual - ephemeris_desired| > tolerance.
    ValueError:
        If ephemeris shapes are not equal.
    ValueError:
        If coordinate dimensions are not one of 2, 3, or 6.
    """
    if ephemeris_type == "cartesian":

        differences, statistics, error = testCartesianEpehemeris(
            ephemeris_actual,
            ephemeris_desired,
            position_tol=position_tol,
            velocity_tol=velocity_tol,
            magnitude=magnitude,
            raise_error=raise_error
        )

    elif ephemeris_type == "spherical":

        differences, statistics, error = testSphericalEpehemeris(
            ephemeris_actual,
            ephemeris_desired,
            position_tol=position_tol,
            velocity_tol=velocity_tol,
            angle_tol=angle_tol,
            angular_velocity_tol=angular_velocity_tol,
            magnitude=magnitude,
            raise_error=raise_error
        )

    else:
        err = ("ephemeris_type should be one of {'cartesian', 'spherical'")
        raise ValueError(err)

    return differences, statistics, error