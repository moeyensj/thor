import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from astroquery.jplhorizons import Horizons

from ....constants import Constants as c
from ..propagate import propagateOrbits

MU = c.G * c.M_SUN
CM = (1.0 * u.cm).to(u.AU).value
MM_P_SEC = (1.0 * u.mm / u.s).to(u.AU / u.d).value

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 
EPOCHS = [57257.0] 
THOR_PROPAGATOR_KWARGS = {
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-15
}
PYOORB_PROPAGATOR_KWARGS = {
    "orbit_type" : "cartesian", 
    "time_scale" : "TT", 
    "magnitude" : 20, 
    "slope" : 0.15, 
    "dynamical_model" : "2",
    "ephemeris_file" : "de430.dat"
}

def test_propagateOrbits():
    # This test makes sure that the universal propagator used by THOR and the 2-body propagator
    # implemented by PYOORB return the same results
    # It is a little shocking that this works as well as it does...
    t0 = Time([epoch for target in TARGETS for epoch in EPOCHS], scale="tdb", format="mjd")
    t1 = Time(np.arange(57999, 57999+50, 0.1), scale="utc", format="mjd")
    
    vectors_list = []
    for name in TARGETS: 
        # Grab vectors from Horizons at epoch
        target = Horizons(id=name, epochs=EPOCHS, location="@sun")
        vectors = target.vectors().to_pandas()
        vectors = vectors[["x", "y", "z", "vx", "vy", "vz"]].values
        vectors_list.append(vectors)
        
    orbits = np.vstack(vectors_list)

    propagated_thor = propagateOrbits(orbits, t0, t1, backend="THOR", backend_kwargs=THOR_PROPAGATOR_KWARGS)
    propagated_pyoorb = propagateOrbits(orbits, t0, t1, backend="PYOORB", backend_kwargs=PYOORB_PROPAGATOR_KWARGS)

    r_diff = np.linalg.norm(propagated_thor[["x", "y", "z"]].values - propagated_pyoorb[["x", "y", "z"]].values, axis=1)
    np.testing.assert_allclose(r_diff, np.zeros(len(r_diff)), atol=CM, rtol=0.0)

    v_diff = np.linalg.norm(propagated_thor[["vx", "vy", "vz"]].values - propagated_pyoorb[["vx", "vy", "vz"]].values, axis=1)
    np.testing.assert_allclose(v_diff, np.zeros(len(v_diff)), atol=MM_P_SEC, rtol=0.0)
    return