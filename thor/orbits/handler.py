from ..constants import Constants as c

__all__ = [
    "_backendHandler"
]

MU = c.G * c.M_SUN

THOR_PROPAGATOR_KWARGS = {
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-15,
    "origin" : "heliocenter"
}

PYOORB_PROPAGATOR_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "TT", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "N",
    "ephemeris_file" : "de430.dat"
}

THOR_EPHEMERIS_KWARGS = {
    "light_time" : True, 
    "lt_tol" : 1e-10,
    "stellar_aberration" : False,
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-16
}

PYOORB_EPHEMERIS_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "UTC", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "N",
    "ephemeris_file" : "de430.dat"
}


def _backendHandler(backend, function_type):
    backend_kwargs = {}

    if backend == "THOR":
        if function_type == "propagate":
            backend_kwargs = THOR_PROPAGATOR_KWARGS
        elif function_type == "ephemeris":
            backend_kwargs = THOR_EPHEMERIS_KWARGS

    elif backend == "PYOORB":
        if function_type == "propagate":
            backend_kwargs = PYOORB_PROPAGATOR_KWARGS
        elif function_type == "ephemeris":
            backend_kwargs = PYOORB_EPHEMERIS_KWARGS

    else:
        err = (
            "backend should be one of 'THOR' or 'PYOORB'"
        )
        raise ValueError(err)

    return backend_kwargs


