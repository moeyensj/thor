import numpy as np
from astropy import units as u
from astropy import constants as c

from .config import Config
from .vectors import calcNae
from .vectors import calcDelta
from .vectors import calcXae
from .vectors import calcXa
from .vectors import calcNhat
from .vectors import calcR1
from .vectors import calcR2
from .projections import cartesianToGnomonic
from .coordinates import transformCoordinates

__all__ = ["TestParticle"]

class TestParticle:
    """
    TestParticle: Class that calculates and stores the rotation matrices 
    for a guess of heliocentric distance and velocity. To be used in 
    tandem with the Cell class.
    
    Parameters
    ----------
    coords_eq_ang : `~numpy.ndarray` (2)
        Angular equatorial coordinates.
    r : float
        Heliocentric distance in AU.
    v : `~numpy.ndarray` (1, 3)
        Particle's velocity vector in AU per day (ecliptic). 
    x_e : `~numpy.ndarray` (1, 3)
        Topocentric position vector in AU (ecliptic). 
    mjd : float
        Time at which the given geometry is true in units of MJD.
    """
    def __init__(self, coords_eq_ang, r, v, x_e, mjd):
        self.coords_eq_ang = coords_eq_ang
        self.r = r
        self.v = v
        self.x_e = x_e
        self.mjd = mjd
        
        
    def prepare(self, verbose=True):
        """
        Calculate rotation matrices. 
        
        Populates the following class properties:
            coords_ec : ecliptic coordinates 
            n_ae : observer to object unit vector 
            delta : observer to object distance assuming r 
            x_ae : observer to object position vector
            x_a : object position vector
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
            print("Preparing rotation matrices...")
            print("Convering to ecliptic coordinates...")
        self.coords_ec = transformCoordinates(self.coords_eq_ang, "equatorial", "ecliptic")
        
        if verbose is True:
            print("Calculating object to observer unit vector...")
        self.n_ae = calcNae(self.coords_ec[:, 0:2])
        
        if verbose is True:
            print("Calculating object to observer distance assuming r = {} AU...".format(self.r))
        self.delta = calcDelta(self.r, self.x_e, self.n_ae)
        
        if verbose is True:
            print("Calculating object to observer position vector...")
        self.x_ae = calcXae(self.delta, self.n_ae)
        
        if verbose is True:
            print("Calculating barycentic object position vector...")
        self.x_a = calcXa(self.x_ae, self.x_e)
        
        if verbose is True:
            print("Creating elements array...")
        self.elements = np.array([*self.x_a, *self.v])
        
        if verbose is True:
            print("Calculating vector normal to plane of orbit...")
        self.n_hat = calcNhat(self.x_a)
        
        if verbose is True:
            print("Calculating R1 rotation matrix...")
        self.R1 = calcR1(self.x_a, self.n_hat)
        self.x_a_xy = np.array(self.R1 @ self.x_a)[0]
        
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
        
    def apply(self, cell, verbose=True, columnMapping=Config.columnMapping):
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
        columnMapping : dict, optional
            Column name mapping of observations to internally used column names. 
            [Default = `~thor.Config.columnMapping`]
        
        Returns
        -------
        None
        """
        
        if verbose is True:
            print("Applying rotation matrices to observations...")
            print("Convering to ecliptic coordinates...")
        coords_ec = equatorialToEclipticAngular(np.radians(cell.observations[[columnMapping["RA_deg"], columnMapping["Dec_deg"]]].values))
        
        if verbose is True:
            print("Calculating object to observer unit vector...")
        n_ae = calcNae(coords_ec)
        x_e = cell.observations[[columnMapping["obs_x_au"], columnMapping["obs_y_au"], columnMapping["obs_z_au"]]].values
        
        if verbose is True:
            print("Calculating object to observer distance assuming r = {} AU...".format(self.r))
        delta = np.zeros(len(n_ae))
        for i, (n_ae_i, x_e_i) in enumerate(zip(n_ae, x_e)):
            delta[i] = calcDelta(self.r, x_e_i, n_ae_i)
        
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