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
        _checkTime(epoch, "epoch")
        
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
        
    def apply(self, cell, verbose=True):
        """
        Apply the prepared rotations to the given cell. Adds the gnomonic 
        plane coordinates to the cell's observations (columns: theta_x_deg, theta_y_deg) 
        
        Parameters
        ----------
        cell : `~thor.cell.Cell`
            THOR cell. 
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
        coords_eq = cell.observations[["RA_deg", "Dec_deg"]].values
        coords_eq = np.hstack([np.ones((len(coords_eq), 1)), coords_eq])        
        coords_ec = transformCoordinates(coords_eq, 
            "equatorial", 
            "ecliptic",
            representation_in="spherical",
            representation_out="spherical")
        
        if verbose is True:
            print("Calculating object to observer unit vector...")
        n_ae = calcNae(coords_ec[:, 1:])
        x_e = cell.observations[["obs_x", "obs_y", "obs_z"]].values
        
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
            print("Calculating barycentic object position vector...")
        x_a = np.zeros([len(x_ae), 3])
        for i, (x_ae_i, x_e_i) in enumerate(zip(x_ae, x_e)):
            x_a[i] = calcXa(x_ae_i, x_e_i)
        
        if verbose is True:
            print("Applying rotation matrix M to barycentric object position vector...")
        coords_cart_rotated = np.array(self.M @ x_a.T).T
        
        if verbose is True:
            print("Performing gnomonic projection...")
        gnomonic = cartesianToGnomonic(coords_cart_rotated)
        
        cell.observations["theta_x_deg"] = np.degrees(gnomonic[:, 0])
        cell.observations["theta_y_deg"] = np.degrees(gnomonic[:, 1])

        if verbose is True:
            print("Done.")
            print("")
        return  