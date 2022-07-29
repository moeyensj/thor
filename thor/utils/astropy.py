import pandas as pd
from copy import deepcopy
from astropy.time import Time

__all__ = [
    "times_from_df",
    "times_to_df",
    "_check_times"
]

def times_from_df(df: pd.DataFrame) -> Time:
    """
    Read times from a `~pandas.DataFrame`.

    Parameters
    ----------
    df : `~pandas.DataFrame`
        DataFrame containing times.

    Returns
    -------
    times : `~astropy.time.core.Time`
        Astropy time object containing times read from dataframe.
    """
    time_col = None
    cols = [f"mjd_{s}" for s in Time.SCALES]
    for col in cols:
        if col in df.columns:
            time_col = col
            format, scale = time_col.split("_")
            break

    times = Time(
        df[time_col].values,
        format=format,
        scale=scale
    )
    return times

def times_to_df(times: Time, time_scale: str = "utc") -> pd.DataFrame:
    """
    Store times as a `~pandas.DataFrame`.

    Parameters
    ----------
    times : `~astropy.time.core.Time`
        Astropy time object.
    time_scale : str
        Store times with this time scale.

    Returns
    -------
    df : `~pandas.DataFrame`
        DataFrame containing times.
    """
    data = {}
    time = deepcopy(times)
    time._set_scale(time_scale)
    data[f"mjd_{time.scale}"] = time.mjd

    return pd.DataFrame(data)

def _check_times(times, arg_name="times"):
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
    if type(times) != Time:
        raise TypeError(err.format(arg_name))
    return
