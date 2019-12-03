import numpy as np
from astroquery.jplhorizons import Horizons

from ....constants import Constants as c
from ....coordinates import _cartesianToAngular
from ....coordinates import _angularToCartesian
from ...propagate import propagateUniversal
from ..gauss import gaussIOD

MU = c.G * c.M_SUN

def test_gaussIOD():
    
    epoch = [58762.0]
    
    #observer = Horizons(id="Ceres", epochs=epoch, location="@sun")
    #vectors = observer.vectors()
    #vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
    #vectors_obs = vectors.reshape(1, -1)
    vectors_obs = np.array([[1., 0., 0., 0., -np.sqrt(MU), 0.]]) 
    
    t0 = np.array(epoch, dtype=float)
    t1 = epoch[0] + np.arange(1, 365*10, 1, dtype=float)
    
    states_observer = propagateUniversal(vectors_obs, t0, t1, mu=MU, max_iter=1000, tol=1e-15)
    states_observer = states_observer[:, 2:]
    
    targets = ["Ceres", "Eros", "1719"]
    
    for target in targets:
        target = Horizons(id=target, epochs=epoch, location="@sun")
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
        vectors_target = vectors.reshape(1, -1)

        states_target = propagateUniversal(vectors_target, t0, t1, mu=MU, max_iter=1000, tol=1e-15)
        states_target = states_target[:, 2:]

        delta = states_target[:, :3] - states_observer[:, :3]
        
        rho = np.zeros_like(delta)
        for i, j in enumerate(delta):
            rho[i] = j / np.linalg.norm(j)

        rho_eq = np.array(c.TRANSFORM_EC2EQ @ rho.T).T
    
        coords_ec = _cartesianToAngular(*rho.T)[:, :2]
        coords_ec = np.degrees(coords_ec)

        coords_eq = _cartesianToAngular(*rho_eq.T)[:, :2]
        coords_eq = np.degrees(coords_eq)

        selected_obs = [0, 4, 8]
        truth_r = states_target[selected_obs, :3]
        truth_v = states_target[selected_obs, 3:]
        coords_obs = states_observer[selected_obs, :3]
        coords_ec_ang = coords_ec[selected_obs]
        coords_eq_ang = coords_eq[selected_obs]
        t = t1[selected_obs]
        
        orbits = gaussIOD(coords_eq_ang, t, coords_obs, velocity_method="gibbs", iterate=True, mu=MU, max_iter=100, tol=1e-15)
        
        closest_r = 1e10
        closest_v = 1e10

        for i, orbit in enumerate(orbits):
            print(orbit)
            
            r2 = orbit[:3]
            v2 = orbit[3:]
            r2_mag = np.linalg.norm(r2)
            v2_mag = np.linalg.norm(v2)

            r2_truth = truth_r[1,:]
            v2_truth = truth_v[1,:]
            r2_truth_mag = np.linalg.norm(r2_truth)
            v2_truth_mag = np.linalg.norm(v2_truth)


            r_diff = (r2_mag - r2_truth_mag) / r2_truth_mag
            v_diff = (v2_mag - v2_truth_mag) / v2_truth_mag
            
            if closest_r > np.abs(r_diff):
                closest_r = r_diff
                closest_v = v_diff
            
        # Test position to within a couple of meters and velocity to within a mm/s
        np.testing.assert_allclose(closest_r, 0.0, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(closest_v, 0.0, atol=1e-10, rtol=1e-10)
                
                