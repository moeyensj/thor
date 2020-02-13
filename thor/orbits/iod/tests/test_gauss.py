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
    
    # Set an initial epoch
    t0 = np.array(epoch, dtype=float)

    # Set propagation epochs: 10 years at 1 day intervals
    t1 = epoch[0] + np.arange(1, 365*10, 1, dtype=float)
    
    # Propagate the observer to each epoch and grab its state
    states_observer = propagateUniversal(vectors_obs, t0, t1, mu=MU, max_iter=1000, tol=1e-15)
    states_observer = states_observer[:, 2:]
    
    # Run IOD on the following targets
    targets = ["Ceres", "Eros", "1719", "Amor"]
    
    for name in targets:
        
        # Grab target's state at t0 from Horizons
        target = Horizons(id=name, epochs=epoch, location="@sun")
        vectors = target.vectors()
        vectors = np.array(vectors["x", "y", "z", "vx", "vy", "vz"]).view("float64")
        vectors_target = vectors.reshape(1, -1)
        
        # Propagate target to each t1 epoch from the initial state
        states_target = propagateUniversal(vectors_target, t0, t1, mu=MU, max_iter=1000, tol=1e-15)
        states_target = states_target[:, 2:]

        # Calculate the distance from observer to target at each epoch
        delta = states_target[:, :3] - states_observer[:, :3]
        
        # Calculate the line of sight vectors at each t1 epoch
        rho = np.zeros_like(delta)
        for i, j in enumerate(delta):
            rho[i] = j / np.linalg.norm(j)

        # Using line of sight vectors, calculate the on-sky location of the target
        # from the point of view of the observer in both ecliptic and equatorial 
        # coordinates
        rho_eq = np.array(c.TRANSFORM_EC2EQ @ rho.T).T
        coords_ec = _cartesianToAngular(*rho.T)[:, :2]
        coords_ec = np.degrees(coords_ec)

        coords_eq = _cartesianToAngular(*rho_eq.T)[:, :2]
        coords_eq = np.degrees(coords_eq)

        # Iterate over different selections of "observations"
        for selected_obs in [[0, 5, 10], 
                             [100, 120, 140],
                             [22, 23, 24],
                             [1000, 1005, 1010],
                             [3600, 3620, 3630],
                             [1999, 2009, 2034]]:

            print("Target Name: {}".format(name))
            print("Observations Indices: \n\t{}".format(selected_obs))

            # Grab truth position and velocity vectors 
            truth_r = states_target[selected_obs, :3]
            truth_v = states_target[selected_obs, 3:]

            # Grab observables: on-sky location of the target
            coords_obs = states_observer[selected_obs, :3]
            coords_ec_ang = coords_ec[selected_obs]
            coords_eq_ang = coords_eq[selected_obs]
            t = t1[selected_obs]

            print("Observations:")
            for i, observation in enumerate(coords_eq_ang):
                print("\tt [MJD]: {}, RA [Deg]: {}, Dec [Deg]: {}, obs_x [AU]: {}, obs_y [AU]: {}, obs_z [AU]: {}".format(t[i], *observation, *coords_obs[i]))

            state_truth = np.concatenate([np.array([t[1]]), states_target[selected_obs[1]]])
            print("Actual State [MJD, AU, AU/d]:\n\t{}".format(state_truth))

            # Using observables, run IOD without iteration
            orbits = gaussIOD(coords_eq_ang, t, coords_obs, velocity_method="gibbs", iterate=False, mu=MU, max_iter=100, tol=1e-15, light_time=False)

            print("Predicted States (without iteration) [MJD, AU, AU/d]:")
            for orbit in orbits:
                print("\t{}".format(orbit))
            
            # Using observables, run IOD
            orbits = gaussIOD(coords_eq_ang, t, coords_obs, velocity_method="gibbs", iterate=True, mu=MU, max_iter=100, tol=1e-15, light_time=False)

            print("Predicted States (with iteration) [MJD, AU, AU/d]:")
            for orbit in orbits:
                print("\t{}".format(orbit))
            
            # IOD returns up to 3 solutions, iterate over each one and find the one with the closest
            # prediction in position, if the position is within 100 meters and the velocity is within 
            # 10 cm/s we are happy
            closest_r = 1e10
            closest_v = 1e10

            for i, orbit in enumerate(orbits):            
                r2 = orbit[1:4]
                v2 = orbit[4:]
                r2_mag = np.linalg.norm(r2)
                v2_mag = np.linalg.norm(v2)

                r2_truth = truth_r[1,:]
                v2_truth = truth_v[1,:]
                r2_truth_mag = np.linalg.norm(r2_truth)
                v2_truth_mag = np.linalg.norm(v2_truth)

                r_diff = np.abs(r2_mag - r2_truth_mag) / r2_truth_mag
                v_diff = np.abs(v2_mag - v2_truth_mag) / v2_truth_mag
                
                if closest_r > np.abs(r_diff):
                    closest_r = r_diff
                    closest_v = v_diff

            print("(Actual - Predicted) / Actual:")
            for orbit in orbits:
                print("\t{}".format((states_target[selected_obs[1]] - orbit[1:]) / states_target[selected_obs[1]]))
            print("")
                
            # Test position to within 100 meters and velocity to within 10 cm/s
            np.testing.assert_allclose(closest_r, 0.0, atol=6.68459e-10, rtol=6.68459e-10)
            np.testing.assert_allclose(closest_v, 0.0, atol=5.77548e-8, rtol=5.77548e-8)
                
                