import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jacfwd,
    vmap
)
from typing import Callable

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

__all__ = [
    "calc_jacobian"
]

def calc_jacobian(
        coords: np.ndarray,
        _func: Callable,
        **kwargs,
    ) -> jnp.ndarray:
    """
    Calculate the jacobian for the given callable in D dimensions for every
    N coordinate.

    Parameters
    ----------
    coords : `~numpy.ndarray` (N, D)
        Coordinates that correspond to the input covariance matrices.
    _func : function
        A function that takes a single coord (D) as input and return the transformed
        coordinate (D). See for example: `thor.coordinates._cartesian_to_spherical`
        or `thor.coordinates._cartesian_to_keplerian`.

    Returns
    -------
    jacobian : `~jax.numpy.ndarray` (N, D, D)
        Array containing function partial derivatives for each coordinate.
    """
    # Calculate the jacobian function for the input function
    # Do this only once!
    jacobian_func = jacfwd(_func, argnums=0)

    in_axes = [0]
    for k, v in kwargs.items():
        if isinstance(v, (np.ndarray, np.ma.masked_array)):
            in_axes.append(0)
        else:
            in_axes.append(None)
    in_axes = tuple(in_axes)

    vmapped_jacobian_func = vmap(
        jacobian_func,
        in_axes=in_axes,
        out_axes=(0),
    )

    jacobian = vmapped_jacobian_func(coords, *kwargs.values())
    return jacobian