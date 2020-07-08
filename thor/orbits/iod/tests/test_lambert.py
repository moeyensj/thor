import numpy as np
from astropy.time import Time
from astropy import units as u

from ....constants import Constants as c
from ....utils import getHorizonsVectors
from ...propagate import propagateOrbits
from ..lambert import calcLambert


MU = c.G * c.M_SUN

TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
    "C/2019 Q4" # Borisov
] 

T0 = Time([58800.0], scale="tdb", format="mjd")
T1 = Time(np.arange(58800.0, 58800.0 + 60, 0.5), scale="tdb", format="mjd")

THOR_PROPAGATOR_KWARGS = {
    "mu" : MU,
    "max_iter" : 1000, 
    "tol" : 1e-15,
    "origin" : "heliocenter",
}


def test_calcLambert():

    for target in TARGETS:
        # Query Horizons for heliocentric geometric states at each T1
        horizons_states = getHorizonsVectors(target, T1, location="@sun", aberrations="geometric")
        horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Propagate the state at T0 to all T1 using THOR 2-body
        thor_states = propagateOrbits(horizons_states[:1], T0, T1, backend="THOR", backend_kwargs=THOR_PROPAGATOR_KWARGS)
        thor_states = thor_states[["x", "y", "z", "vx", "vy", "vz"]].values
        
        for selected_obs in [[0, 1], [23, 52], [0, -1]]:
            
            r0 = thor_states[selected_obs[0], :3]
            t0 = T1[selected_obs[0]].utc.mjd
            r1 = thor_states[selected_obs[1], :3]
            t1 = T1[selected_obs[1]].utc.mjd
            
            v0, v1 = calcLambert(r0, t0, r1, t1, mu=MU, max_iter=1000, dt_tol=1e-12)
        
            v0_diff = np.linalg.norm(thor_states[selected_obs[0], 3:] - v0) * (u.AU / u.d).to(u.cm / u.s)  
            v1_diff = np.linalg.norm(thor_states[selected_obs[1], 3:] - v1) * (u.AU / u.d).to(u.cm / u.s)  

            np.testing.assert_allclose(np.abs(v0_diff), np.zeros(1), atol=5, rtol=0)
            np.testing.assert_allclose(np.abs(v1_diff), np.zeros(1), atol=5, rtol=0)
            
    return
