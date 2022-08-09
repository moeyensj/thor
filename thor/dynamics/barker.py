import jax.numpy as jnp
from jax import (
    config,
    jit
)

config.update("jax_enable_x64", True)

__all__ = [
    "solve_barker"
]

@jit
def solve_barker(M: float) -> float:
    """
    Solve Barker's equation for true anomaly given parabolic mean
    anomaly.

    Parameters
    ----------
    M : float
        Parabolic mean anomaly (equal to sqrt(mu / (2 q^3))(t0 - tp)).

    Returns
    -------
    nu : float
        True anomaly in radians.

    References
    ----------
    [1] Curtis, H. D. (2014). Orbital Mechanics for Engineering Students. 3rd ed.,
        Elsevier Ltd. ISBN-13: 978-0080977478
    """
    # Equation 3.32 in Curtis (2014) [1]
    nu = 2 * jnp.arctan(
        (3 * M + jnp.sqrt((3 * M)**2 + 1))**(1/3)
        - (3 * M + jnp.sqrt((3 * M)**2 + 1))**(-1/3)
    )
    return nu