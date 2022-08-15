import numpy as np
import jax.numpy as jnp
from jax import (
    config,
    jacfwd,
    vmap
)
from typing import (
    Callable,
    Hashable,
    Tuple,
    Optional,
)

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

__all__ = [
    "calc_jacobian"
]

def calc_jacobian(
        coords: np.ndarray,
        _func: Callable,
        in_axes: Optional[Hashable] = (0,),
        out_axes: Optional[int] = 0,
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
    in_axes : Optional[Hashable]
        An integer or ``None`` indicates which array axis to map over for all arguments (with ``None``
        indicating not to map any axis), and a tuple indicates which axis to map
        for each corresponding positional argument. Axis integers must be in the
        range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
        dimensions (axes) of the corresponding input array.

        From: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap
    out_axes : Optional[int]
        An integer, None, or (nested) standard Python container (tuple/list/dict) thereof
        indicating where the mapped axis should appear in the output. All outputs with a
        mapped axis must have a non-None out_axes specification.

        From: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap

    Returns
    -------
    jacobian : `~numpy.ndarray` (N, D, D)
        Array containing function partial derivatives for each coordinate.
    """
    # Calculate the jacobian function for the input function
    # Do this only once!
    jacobian_func = jacfwd(_func, argnums=0)

    vmapped_jacobian_func = vmap(
        jacobian_func,
        in_axes=in_axes,
        out_axes=out_axes,
    )

    jacobian = vmapped_jacobian_func(coords, *kwargs.values())
    # If the vmapped function returns more outputs, then only
    # return the first one. All relevant functions in THOR return
    # primary result first, though we may want to come up with a more general
    # solution in the future.
    if isinstance(jacobian, Tuple):
        jacobian = jacobian[0]
    return np.array(jacobian)