import numpy as np
from astropy import units as u
from astropy import constants as c

from .vectors import calcNae
from .vectors import calcDelta
from .vectors import calcXae
from .vectors import calcXa
from .vectors import calcNhat
from .vectors import calcR1
from .vectors import calcR2
from .projections import cartesianToGnomonic
from .coordinates import equatorialToEclipticAngular

__all__ = ["TestParticle"]

class TestParticle:
    def __init__(self, coords_eq_ang, r, velocity_ec_cart, x_e, mjd=None):
        self.coords_eq_ang = coords_eq_ang
        self.r = r
        self.velocity_ec_cart = velocity_ec_cart
        self.x_e = x_e
        self.mjd = mjd
        
        
    def prepare(self, verbose=True):
        if verbose is True:
            print("Convering to ecliptic coordinates...")
        self.coords_ec = equatorialToEclipticAngular(np.radians([self.coords_eq_ang]))
        
        if verbose is True:
            print("Calculating asteroid to observer unit vector...")
        self.n_ae = calcNae(self.coords_ec[:, 0:2])
        
        if verbose is True:
            print("Calculating asteroid to observer distance assuming r = {} AU...".format(self.r))
        self.delta = calcDelta(self.r, self.x_e, self.n_ae)
        
        if verbose is True:
            print("Calculating asteroid to observer position vector...")
        self.x_ae = calcXae(self.delta, self.n_ae)
        
        if verbose is True:
            print("Calculating barycentic asteroid position vector...")
        self.x_a = calcXa(self.x_ae, self.x_e)
        
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
        
    def apply(self, cell, verbose=True):
        
        if verbose is True:
            print("Convering to ecliptic coordinates...")
        coords_ec = equatorialToEclipticAngular(np.radians(cell.observations[["RA_deg", "Dec_deg"]].values))
        
        if verbose is True:
            print("Calculating asteroid to observer unit vector...")
        n_ae = calcNae(coords_ec)
        x_e = cell.observations[["obs_x_au", "obs_y_au", "obs_z_au"]].values
        
        if verbose is True:
            print("Calculating asteroid to observer distance assuming r = {} AU...".format(self.r))
        delta = np.zeros(len(n_ae))
        for i, (n_ae_i, x_e_i) in enumerate(zip(n_ae, x_e)):
            delta[i] = calcDelta(self.r, x_e_i, n_ae_i)
        
        if verbose is True:
            print("Calculating asteroid to observer position vector...")
        x_ae = np.zeros([len(delta), 3])
        for i, (delta_i, n_ae_i) in enumerate(zip(delta, n_ae)):
            x_ae[i] = calcXae(delta_i, n_ae_i)
        
        if verbose is True:
            print("Calculating barycentic asteroid position vector...")
        x_a = np.zeros([len(x_ae), 3])
        for i, (x_ae_i, x_e_i) in enumerate(zip(x_ae, x_e)):
            x_a[i] = calcXa(x_ae_i, x_e_i)
        
        if verbose is True:
            print("Applying rotation matrix M to barycentric asteroid position vector...")
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