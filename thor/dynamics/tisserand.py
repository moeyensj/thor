import numpy as np

__all__ = [
    "calc_tisserand_parameter"
]

# This code generates the dictionary of semi-major axes for the
# third body needed for the Tisserand parameter
#
#
# from astropy.time import Time
# from thor.utils import getHorizonsElements
#
# ids = ["199", "299", "399", "499", "599", "699", "799", "899"]
# elements = getHorizonsElements(ids, times, id_type="majorbody")
#
# MAJOR_BODIES = {}
# for i, r in elements[["targetname", "a"]].iterrows():
#    body_name = r["targetname"].split(" ")[0].lower()
#    MAJOR_BODIES[body_name] = r["a"]
#

MAJOR_BODIES = {
    'mercury': 0.3870970330236769,
    'venus': 0.723341974974844,
    'earth': 0.9997889954736553,
    'mars': 1.523803685638066,
    'jupiter': 5.203719697535582,
    'saturn': 9.579110220472034,
    'uranus': 19.18646168457971,
    'neptune': 30.22486701698071
}

def calc_tisserand_parameter(a, e, i, third_body="jupiter"):
    """
    Calculate Tisserand's parameter used to identify potential comets.
    For example, objects with Tisserand parameter's with respect to Jupiter greater than 3 are
    typically asteroids, whereas Jupiter family comets may have Tisserand's parameter
    between 2 and 3. Damocloids have Jupiter Tisserand's parameter of less than 2.

    Parameters
    ----------
    a : float or `~numpy.ndarray` (N)
        Semi-major axis in au.
    e : float or `~numpy.ndarray` (N)
        Eccentricity.
    i : float or `~numpy.ndarray` (N)
        Inclination in degrees.
    third_body : str
        Name of planet with respect to which Tisserand's parameter
        should be calculated.

    Returns
    -------
    Tp : float or `~numpy.ndarray` (N)
        Tisserand's parameter.
    """
    i_rad = np.radians(i)

    major_bodies = MAJOR_BODIES.keys()
    if third_body not in major_bodies:
        err = (
            f"third_body should be one of {','.join(major_bodies)}"
        )
        raise ValueError(err)

    ap = MAJOR_BODIES[third_body]
    Tp = ap / a + 2 * np.cos(i_rad) * np.sqrt(a / ap * (1 - e**2))

    return Tp

