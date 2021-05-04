import numpy as np

from .utils import _checkTime
from .vectors import calcNae
from .vectors import calcDelta
from .vectors import calcXae
from .vectors import calcXa
from .vectors import calcNhat
from .vectors import calcR1
from .vectors import calcR2
from .projections import cartesianToGnomonic
from .coordinates import transformCoordinates

__all__ = ["TestOrbit"]

class TestOrbit:
    """
    TestOrbit: Class that calculates and stores the rotation matrices 
    for a guess of heliocentric distance and velocity. To be used in 
    tandem with the Cell class.
    
    Parameters
    ----------
    elements : `~numpy.ndarray` (6)
        Cartesian ecliptic orbital elements with postions in units of AU
        and velocities in units of AU per day. 
    t0 : `~astropy.time.core.Time` (1)
        Epoch at which orbital elements are defined.
    """
    def __init__(self, elements, epoch):
        self.elements = elements
        self.epoch = epoch
        
    def prepare(self, verbose=True):
        """
        Calculate rotation matrices. 
        
        Populates the following class properties:
            n_hat : vector normal to the plane of orbit 
            R1 : rotation matrix to rotate towards x-y plane
            R2 : rotation matrix to rotate towards x-axis
            M : final rotation matrix
        
        Parameters
        ----------
        verbose : bool, optional
            Print progress statements.
            [Default = True]
            
        Returns
        -------
        None
        """
        if verbose is True:
            print("Calculating vector normal to plane of orbit...")
        self.n_hat = calcNhat(self.elements[:3])
        
        if verbose is True:
            print("Calculating R1 rotation matrix...")
        self.R1 = calcR1(self.elements[:3], self.n_hat)
        self.x_a_xy = np.array(self.R1 @ self.elements[:3])[0]
        
        if verbose is True:
            print("Calculating R2 rotation matrix...")
        self.R2 = calcR2(self.x_a_xy)
        
        if verbose is True:
            print("Calculating final rotation matrix...")
        self.M = self.R2 @ self.R1
        
        if verbose is True:
            print("Done.")
            print("")
        return
        
    def applyToObservations(self, observations, verbose=True):
        """
        Apply the prepared rotations to the given observations. Adds the gnomonic 
        plane coordinates to observations (columns: theta_x_deg, theta_y_deg) 
        
        Parameters
        ----------
        observations : `~pandas.DataFrame`
            DataFrame of observations defined at the same epoch as this test orbit, 
            to project into the test orbit's frame.
        verbose : bool, optional
            Print progress statements? 
            [Default = True]
        
        Returns
        -------
        None
        """
        
        if verbose is True:
            print("Applying rotation matrices to observations...")
            print("Converting to ecliptic coordinates...")

        #velocities_present = False
        #if "vRAcosDec" in observations.columns and "vDec" in observations.columns:
        #    coords_eq_r = observations[["RA_deg", "Dec_deg"]].values
        #    coords_eq_v = observations[["vRAcosDec", "vDec"]].values
        #    coords_eq_v[:, 0] /= np.cos(np.radians(coords_eq_r[:, 1]))
        #    coords_eq = np.hstack([
        #        np.ones((len(coords_eq_r), 1)), 
        #        coords_eq_r, 
        #        np.zeros((len(coords_eq_r), 1)),
        #        coords_eq_v
        #    ])     
        #    velocities_present = True

        #else:
        coords_eq = observations[["RA_deg", "Dec_deg"]].values
        coords_eq = np.hstack([np.ones((len(coords_eq), 1)), coords_eq])        
        coords_ec = transformCoordinates(coords_eq, 
            "equatorial", 
            "ecliptic",
            representation_in="spherical",
            representation_out="spherical"
        )
        
        if verbose is True:
            print("Calculating object to observer unit vector...")
        n_ae = calcNae(coords_ec[:, 1:3])
        x_e = observations[["obs_x", "obs_y", "obs_z"]].values
        
        if verbose is True:
            print("Calculating object to observer distance assuming r = {} AU...".format(np.linalg.norm(self.elements[:3])))
        delta = np.zeros(len(n_ae))
        for i in range(len(delta)):
            delta[i] = calcDelta(np.linalg.norm(self.elements[:3]), x_e[i, :], n_ae[i, :])
        
        if verbose is True:
            print("Calculating object to observer position vector...")
        x_ae = np.zeros([len(delta), 3])
        for i, (delta_i, n_ae_i) in enumerate(zip(delta, n_ae)):
            x_ae[i] = calcXae(delta_i, n_ae_i)
        
        if verbose is True:
            print("Calculating heliocentric object position vector...")
        x_a = np.zeros([len(x_ae), 3])
        for i, (x_ae_i, x_e_i) in enumerate(zip(x_ae, x_e)):
            x_a[i] = calcXa(x_ae_i, x_e_i)
        
        if verbose is True:
            print("Applying rotation matrix M to heliocentric object position vector...")
        coords_cart_rotated = np.array(self.M @ x_a.T).T
        
        if verbose is True:
            print("Performing gnomonic projection...")
        gnomonic_coords = cartesianToGnomonic(coords_cart_rotated)
        

        observations["obj_x"] = x_a[:, 0]
        observations["obj_y"] = x_a[:, 1]
        observations["obj_z"] = x_a[:, 2]
        observations["theta_x_deg"] = np.degrees(gnomonic_coords[:, 0])
        observations["theta_y_deg"] = np.degrees(gnomonic_coords[:, 1])
        observations["test_obj_x"] = self.elements[0]
        observations["test_obj_y"] = self.elements[1]
        observations["test_obj_z"] = self.elements[2]
        observations["test_obj_vx"] = self.elements[3]
        observations["test_obj_vy"] = self.elements[4]
        observations["test_obj_vz"] = self.elements[5]

        if verbose is True:
            print("Done.")
            print("")
        return  

    def applyToEphemeris(self, ephemeris, verbose=True):
        """
        Apply the prepared rotations to the given ephemerides. Adds the gnomonic 
        plane coordinates to observations (columns: theta_x_deg, theta_y_deg, vtheta_x, and vtheta_y) 
        
        Parameters
        ----------
        ephemeris : `~pandas.DataFrame`
            DataFrame of ephemeris generated by a THOR backend defined at the same epoch as this test orbit, 
            to project into the test orbit's frame.
        verbose : bool, optional
            Print progress statements? 
            [Default = True]
        
        Returns
        -------
        None
        """
        coords_cart = ephemeris[["obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz"]].values
        coords_cart_rotated = np.zeros_like(coords_cart)
        
        if verbose is True:
            print("Applying rotation matrix M to heliocentric object position vector...")
        coords_cart_rotated[:, :3] = np.array(self.M @ coords_cart[:, :3].T).T

        if verbose is True:
            print("Applying rotation matrix M to heliocentric object velocity vector...")
        # Calculate relative velocity, then rotate to projected frame
        coords_cart[:, 3:] = coords_cart[:, 3:] - self.elements[3:].reshape(1, -1)
        coords_cart_rotated[:, 3:] = np.array(self.M @ coords_cart[:, 3:].T).T
        
        if verbose is True:
            print("Performing gnomonic projection...")
        gnomonic_coords = cartesianToGnomonic(coords_cart_rotated)
        
        ephemeris["theta_x_deg"] = np.degrees(gnomonic_coords[:, 0])
        ephemeris["theta_y_deg"] = np.degrees(gnomonic_coords[:, 1])
        ephemeris["vtheta_x_deg"] = np.degrees(gnomonic_coords[:, 2])
        ephemeris["vtheta_y_deg"] = np.degrees(gnomonic_coords[:, 3])
        ephemeris["test_obj_x"] = self.elements[0]
        ephemeris["test_obj_y"] = self.elements[1]
        ephemeris["test_obj_z"] = self.elements[2]
        ephemeris["test_obj_vx"] = self.elements[3]
        ephemeris["test_obj_vy"] = self.elements[4]
        ephemeris["test_obj_vz"] = self.elements[5]

        if verbose is True:
            print("Done.")
            print("")
        return 