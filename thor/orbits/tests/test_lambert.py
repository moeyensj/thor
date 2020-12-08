import numpy as np
from astropy.time import Time
from astropy import units as u

from ....constants import Constants as c
from ...utils import getHorizonsVectors
from ...testing import testOrbits
from ..propagate import propagateOrbits
from ..lambert import calcLambert

MU = c.G * c.M_SUN
TARGETS = [
    "Amor",
    "Eros", 
    "Eugenia",
] 
EPOCH = 57257.0
DT = np.arange(0, 14)
T0 = Time(
    [EPOCH for i in range(len(TARGETS))],
    format="mjd",
    scale="tdb", 
)
T1 = Time(
    EPOCH + DT, 
    format="mjd",
    scale="tdb"
)
OBSERVATORIES = ["I11", "I41", "500"]


def test_calcLambert():

    for target in TARGETS:
        # Query Horizons for heliocentric geometric states at each T1
        horizons_states = getHorizonsVectors(
            [target], 
            T1, 
            location="@sun", 
            aberrations="geometric"
            )
        horizons_states = horizons_states[["x", "y", "z", "vx", "vy", "vz"]].values

        # Propagate the state at T0 to all T1 using THOR 2-body
        thor_states = propagateOrbits(
            horizons_states[:1], 
            T0, 
            T1, 
            backend="THOR",
        )
        thor_states = thor_states[["x", "y", "z", "vx", "vy", "vz"]].values
        
        for selected_obs in [[0, 1]]:
            
            r0 = thor_states[selected_obs[0], :3]
            t0 = T1[selected_obs[0]].utc.mjd
            r1 = thor_states[selected_obs[1], :3]
            t1 = T1[selected_obs[1]].utc.mjd
            
            v0, v1 = calcLambert(
                r0, 
                t0, 
                r1, 
                t1, 
                mu=MU,
                max_iter=1000, 
                dt_tol=1e-12
            )
            lambert_state0 = np.concatenate([r0, v0])
            lambert_state1 = np.concatenate([r1, v1])
        
            # Test that the resulting orbit is within the tolerances of the 
            # true state below
            testOrbits(
                lambert_state0.reshape(1, -1),
                thor_states[selected_obs[0]:selected_obs[0]+1],
                position_tol=(1e-10*u.mm),
                velocity_tol=(10*u.mm/u.s)
            )
            testOrbits(
                lambert_state1.reshape(1, -1),
                thor_states[selected_obs[1]:selected_obs[1]+1],
                position_tol=(1e-10*u.mm),
                velocity_tol=(10*u.mm/u.s)
            )
            
    return
