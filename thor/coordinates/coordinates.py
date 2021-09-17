import numpy as np
from typing import Optional, Union

__all__ = ["_ingest_coordinate"]


def _ingest_coordinate(
    q: Union[list, np.ndarray], d: int, coords: Optional[np.ma.array] = None
):
    """
    Ingest coordinates along an axis (like the x, y, z) and add them to an existing masked array
    of coordinate measurements if that object already exists. If that object doesn't exist then
    create it and return it. Any missing values in q should be represented with NaNs.

    Parameters
    ----------
    q : list or `~numpy.ndarray` (N)
        List or 1-D array of coordinate measurements.
    d : int
        The coordinate axis (as an index). For example, for a 6D Cartesian
        state vector, the x-axis takes the 0th index, the y-axis takes the 1st index,
        the z axis takes the 2nd index, the x-velocity takes the 3rd index, etc..
    coords : `~numpy.ma.ndarray` (N, D), optional
        If coordinates (ie, other axes) have already been defined then pass them here
        so that current axis of coordinates can be added.

    Returns
    -------
    coords : `~numpy.ma.array` (N, 6)
        Masked array of 6D coordinate measurements with q measurements ingested.

    Raises
    ------
    ValueError
        If the length of q doesn't match the length of coords.
    """
    if q is not None:
        q_ = np.asarray(q)
        N_ = len(q_)
        if coords is None:
            coords = np.ma.zeros((N_, 6), fill_value=np.NaN)
            coords.mask = 1
        else:
            N, D = coords.shape
            if N != N_:
                err = (
                    "q needs to be the same length as the existing coordinates.\n"
                    f"q has length {N_} while coords has {N} coordinates in 6 dimensions."
                )
                raise ValueError(err)

        coords[:, d] = q_
        coords.mask[:, d] = np.where(np.isnan(q_), 1, 0)

    return coords
