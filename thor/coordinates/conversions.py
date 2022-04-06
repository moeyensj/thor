import numpy as np

__all__ = [
    "_convert_coordinates_units",
    "_convert_covariances_units"
]

def _convert_coordinates_units(
        coords,
        units,
        desired_units
    ):
    """
    Convert coordinate units to desired units.

    Parameters
    ----------
    coords : `~numpy.ndarray` or `~numpy.ma.core.MaskedArray` (N, D)
        Coordinates that need to be converted.
    units : `~numpy.ndarray` (D)
        Current units for each coordinate dimension.
    desired_units : `~numpy.ndarray` (D)
        Desired units for each coordinate dimension.

    Returns
    -------
    coords_converted : `~numpy.ndarray` or `~numpy.ma.core.MaskedArray` (N, D)
        Coordinates converted to the desired coordinate units.

    Raises
    ------
    ValueError : If units or desired_units do not have length D.
    """
    N, D = coords.shape
    coords_converted = coords.copy()

    if (len(units) != D) or (len(desired_units) != D):
        err = (
            f"Length of units or desired_units does not match the number of coordinate dimensions (D={D})"
        )
        raise ValueError(err)

    for i in range(D):
        coords_converted[:, i] = coords[:, i] * units[i].to(desired_units[i])

    return coords_converted

def _convert_covariances_units(
        covariances,
        units,
        desired_units,
    ):
    """
    Convert covariance units to desired units.

    Parameters
    ----------
    covariances : `~numpy.ndarray` or `~numpy.ma.core.MaskedArray` (N, D, D)
        Covariances that need to be converted.
    units : `~numpy.ndarray` (D)
        Current units for each coordinate dimension. Note, these are not the units for
        the elements of the covariance matrices but the actual units of the mean values/states to
        which the covariance matrices belong.
    desired_units : `~numpy.ndarray` (D)
        Desired units for each coordinate dimension.

    Returns
    -------
    covariances_converted : `~numpy.ndarray` or `~numpy.ma.core.MaskedArray` (N, D, D)
        Covariances converted to the desired coordinate units.

    Raises
    ------
    ValueError : If units or desired_units do not have length D.
    """
    N, D, D = covariances.shape

    if (len(units) != D) or (len(desired_units) != D):
        err = (
            f"Length of units or desired_units does not match the number of coordinate dimensions (D={D})"
        )
        raise ValueError(err)

    coords_units_2d = units.copy().reshape(1, -1)
    covariance_units = np.dot(
        coords_units_2d.T,
        coords_units_2d
    )

    desired_units_2d = desired_units.copy().reshape(1, -1)
    desired_units = np.dot(
        desired_units_2d.T,
        desired_units_2d
    )

    conversion_matrix = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        for j in range(D):
            conversion_matrix[i, j] = covariance_units[i, j].to(desired_units[i, j])

    covariances_converted = conversion_matrix * covariances

    return covariances_converted