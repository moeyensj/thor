from astropy.time import Time

__all__ = ["_checkTime"]


def _checkTime(time, arg_name):
    """
    Check that 'time' is an astropy time object, if not, raise an error.

    Parameters
    ----------
    time : `~astropy.time.core.Time`

    arg_name : str
        Name of argument in function.

    Returns
    -------
    None

    Raises
    ------
    ValueError : If time is not an astropy time object.
    """
    err = (
        "Time ({}) has to be an `~astropy.time.core.Time` object.\n"
        "Convert using:\n\n"
        "from astropy.time import Time\n"
        "times = Time(t_array, scale='...', format='...')"
    )
    if type(time) != Time:
        raise TypeError(err.format(arg_name))
    return
