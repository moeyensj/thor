import numpy as np
from astropy import units as u

__all__ = [
    "testCartesianOrbits",
    "testKeplerianOrbits",
    "testCometaryOrbits",
    "testOrbits",
]

def __statsToErrorMessage(stats, message):
    """
    Helper function that adds the stats derived from _evaluateDifference
    to an error message.
    """
    for stat, value in stats.items():
        message += ("  {:<7}: {}\n".format(stat, value.T))
    message += "\n"
    return message

def _evaluateDifference(
        actual, 
        desired, 
        unit, 
        tol, 
        magnitude=False
    ):
    """
    Calculate the absolute difference between actual and desired
    and return it in the same units of the given tolerance with
    some summary statistics on the absolute difference.
    
    
    Paramters
    ---------
    actual : `~numpy.ndarray` (N, M)
        Array of computed values.
    desired : `~numpy.ndarray` (N, M)
        Array of desired values.
    unit : `astropy.units.core.Unit`
        Units in which both arrays are expressed. Must be the same.
    tol : `~astropy.units.quantity.Quantity` (1)
        The tolerance to which the absolute difference will be evaluated 
        to. Used purely to convert the absolute differences to the same
        units as the tolerance.
    
    Returns
    -------
    diff : `~astropy.units.quantity.Quantity` (N, M)
        |actual - desired| in units of the given tolerance.
    stats : dict
        "Mean" : mean difference per M dimension
        "Median" : median difference per M dimension
        "Min" : minimum difference per M dimension 
        "Max" : maximum difference per M dimension
        "Argmin" : location of minimum difference per M dimension
        "Argmax" : location of maximum difference per M dimension
    error : bool
        True if diff is not within the desired tolerance. 
    """
    error = False
    diff = np.abs(actual - desired) * unit
    diff = diff.to(tol.unit)
   
    if magnitude:
        diff = np.linalg.norm(diff, axis=1)
        diff = diff[:, np.newaxis]

    tol_ = np.empty_like(diff)
    tol_.fill(tol)
    
    stats = {
        "Mean" : np.mean(diff, axis=0),
        "Median" : np.median(diff, axis=0),
        "Std" : np.std(diff, axis=0),
        "Min" : np.min(diff, axis=0),
        "Max" : np.max(diff, axis=0),
        "Argmin" : np.argmin(diff, axis=0),
        "Argmax" : np.argmax(diff, axis=0)
    }

    try:
        np.testing.assert_array_less(
            diff.to(tol.unit), 
            tol_.to(tol.unit), 
        )
    except AssertionError as e:
        error = True
    return diff, stats, error

def testCartesianOrbits(
        orbits_actual, 
        orbits_desired,
        position_tol=1*u.m, 
        velocity_tol=(1*u.mm/u.s),
        magnitude=True,
        raise_error=True
    ):
    """
    Tests that the two sets of cartesian orbits are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|. 
    
    Parameters
    ----------
    orbits_actual : `~numpy.ndarray` (N, 6)
        Array of orbits to compare to the desired orbits. 
        Assumed units for:
            positions : AU,
            velocities : AU per day
    orbits_desired : `~numpy.ndarray` (N, 6)
        Array of desired orbits to which to compare the actual orbits to. 
        Assumed units for:
            positions : AU,
            velocities : AU per day
    orbit_type : {'cartesian', 'keplerian', 'cometary'}
        Type of the input orbits. Both actual and desired orbits must be of the same
        type. 
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
        If |orbits_actual - orbits_desired| > tolerance. 
        
    Returns
    -------
    None
    """
    any_error = False
    error_message = "\n"
    differences = {}
    statistics = {}
    
    # Test positions
    if magnitude:
        names = ["r"]
    else:
        names = ["x", "y", "z"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, :3], 
        orbits_desired[:, :3], 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), position_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    # Test velocities
    if magnitude:
        names = ["v"]
    else:
        names = ["vx", "vy", "vz"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 3:], 
        orbits_desired[:, 3:], 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), velocity_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    if any_error and raise_error:
        raise AssertionError(error_message)
        
    return differences, statistics, error

def testKeplerianOrbits(
        orbits_actual, 
        orbits_desired,
        position_tol=1*u.m, 
        unitless_tol=1e-10*u.dimensionless_unscaled,
        angle_tol=1e-10*u.degree,
        raise_error=True
    ):
    """
    Tests that the two sets of keplerian orbits are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|. 
    
    Parameters
    ----------
    orbits_actual : `~numpy.ndarray` (N, 6)
        Array of orbits to compare to the desired orbits. 
        Assumed units for:
            positions : AU,
            angles : degrees
    orbits_desired : `~numpy.ndarray` (N, 6)
        Array of desired orbits to which to compare the actual orbits to. 
        Assumed units for:
            positions : AU,
            angles : degrees
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (a).
    unitless_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance unitless quantities need to satisfy (e).
    angle_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angle quantities need to satisfy (i, ascNode, argPeri, meanAnom).
        
    Raises
    ------
    AssertionError:
        If |orbits_actual - orbits_desired| > tolerance. 
    
    Returns
    -------
    None
    """
    any_error = False
    error_message = "\n"
    differences = {}
    statistics = {}
    
    # Test positions
    names = ["a"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, :1], 
        orbits_desired[:, :1], 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), position_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    # Test unitless (eccentricity)
    names = ["e"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 1:2], 
        orbits_desired[:, 1:2], 
        u.dimensionless_unscaled, 
        unitless_tol, 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), unitless_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
   
    # Test angles   
    names = ["i", "ascNode", "argPeri", "meanAnom"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 2:], 
        orbits_desired[:, 2:], 
        u.degree, 
        angle_tol, 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), angle_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    if any_error and raise_error:
        raise AssertionError(error_message)
        
    return diff, stats, error

def testCometaryOrbits(
        orbits_actual, 
        orbits_desired,
        position_tol=1*u.m, 
        unitless_tol=1e-10*u.dimensionless_unscaled,
        angle_tol=1e-10*u.degree,
        time_tol=1e-6*u.s,
        raise_error=True
    ):
    """
    Tests that the two sets of cometary orbits are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|. 
    
    Parameters
    ----------
    orbits_actual : `~numpy.ndarray` (N, 6)
        Array of orbits to compare to the desired orbits. 
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            times : days
    orbits_desired : `~numpy.ndarray` (N, 6)
        Array of desired orbits to which to compare the actual orbits to. 
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            times : days
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (q).
    unitless_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance unitless quantities need to satisfy (e).
    angle_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angle quantities need to satisfy (i, ascNode, argPeri).
    time_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance time quantities need to satisfy (tPeri).
        
    Raises
    ------
    AssertionError:
        If |orbits_actual - orbits_desired| > tolerance. 
        
    Returns
    -------
    None
    """
    any_error = False
    error_message = "\n"
    differences = {}
    statistics = {}
    
    # Test positions
    names = ["q"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, :1], 
        orbits_desired[:, :1], 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), position_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    # Test unitless (eccentricity)
    names = ["e"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 1:2], 
        orbits_desired[:, 1:2], 
        u.dimensionless_unscaled, 
        unitless_tol, 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), unitless_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
   
    # Test angles 
    names = ["i", "ascNode", "argPeri"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 2:5], 
        orbits_desired[:, 2:5], 
        u.degree, 
        angle_tol, 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), angle_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
        
    
    # Test time (time of perihelion passage)
    names = ["tPeri"]
    diff, stats, error = _evaluateDifference(
        orbits_actual[:, 5:], 
        orbits_desired[:, 5:], 
        u.d, 
        time_tol, 
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
        error_message += "{} difference (|actual - desired|) is not within {}.\n".format(", ".join(names), time_tol)
        error_message = __statsToErrorMessage(
            stats, 
            error_message
        )
    
    if any_error and raise_error:
        raise AssertionError(error_message)
        
    return diff, stats, error

def testOrbits(
        orbits_actual,
        orbits_desired,
        orbit_type="cartesian",
        position_tol=1*u.m, 
        velocity_tol=(1*u.m / u.s),
        unitless_tol=1e-10*u.dimensionless_unscaled,
        angle_tol=1e-10*u.degree,
        time_tol=1e-6*u.s,
        magnitude=True,
        raise_error=False
    ):
    """
    Tests that the two sets of orbits are within the desired absolute tolerances
    of each other. The absolute difference is calculated as |actual - desired|. 
    
    Parameters
    ----------
    orbits_actual : `~numpy.ndarray` (N, 6)
        Array of orbits to compare to the desired orbits. 
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            times : days
    orbits_desired : `~numpy.ndarray` (N, 6)
        Array of desired orbits to which to compare the actual orbits to. 
        Assumed units for:
            positions : AU,
            velocities : AU per day,
            angles : degrees,
            times : days
    orbit_type : {'cartesian', 'keplerian', 'cometary'}
        Type of the input orbits. Both actual and desired orbits must be of the same
        type. 
    position_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance positions need to satisfy (a, q, x, y, z, r).
    velocity_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance velocity need to satisfy. (vx, vy, vz, v).
    unitless_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance unitless quantities need to satisfy (e).
    angle_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance angle quantities need to satisfy (i, ascNode, argPeri, meanAnom).
    time_tol : `~astropy.units.quantity.Quantity` (1)
        Absolute tolerance time quantities need to satisfy (tPeri).
    magnitude : bool
        Applies to cartesian orbits only. Test the magnitude of the position difference
        and velocity difference vectors as opposed to testing per individual coordinate. 
        
    Raises
    ------
    AssertionError:
        If |orbits_actual - orbits_desired| > tolerance. 
    ValueError:
        If orbit_type is not one of 'cartesian', 'keplerian' or 'cometary'.
        
    Returns
    -------
    None
    """
    if orbit_type == "cartesian":

        diff, stats, error = testCartesianOrbits(
            orbits_actual, 
            orbits_desired,
            position_tol=position_tol, 
            velocity_tol=velocity_tol,
            magnitude=magnitude,
            raise_error=raise_error
        )
        
    elif orbit_type == "keplerian":

        diff, stats, error = testKeplerianOrbits(
            orbits_actual, 
            orbits_desired,
            position_tol=position_tol, 
            unitless_tol=unitless_tol,
            angle_tol=angle_tol,
            raise_error=raise_error
        )
       
    elif orbit_type == "cometary":
        
        diff, stats, error = testCometaryOrbits(
            orbits_actual, 
            orbits_desired,
            position_tol=position_tol, 
            unitless_tol=unitless_tol,
            angle_tol=angle_tol,
            time_tol=time_tol,
            raise_error=raise_error
        )
        
    else:
        err = "orbit_type should be one of {'cartesian', 'keplerian', 'cometary'}"
        raise ValueError(err)

    return diff, stats, error
    