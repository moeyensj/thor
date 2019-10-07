import numpy as np
from astropy import units as u
from astropy.constants import codata2014 as c

class Constants:
    
    # Speed of light: AU per day (173.14463267424034)
    C = c.c.to(u.au / u.d).value 
    
    # Gravitational constant: AU**3 / M_sun / d**2 (0.00029591220819207784)
    G = c.G.to(u.AU**3 / u.M_sun / u.d**2).value 
    
    # Solar Mass: M_sun (1.0)
    M_SUN = 1.0
    
    # Earth Mass: M_sun (3.0034893488507934e-06)
    M_EARTH = u.M_earth.to(u.M_sun)
    
    # Mean Obliquity at J2000: radians (0.40909280422232897)
    OBLIQUITY = (84381.448 * u.arcsecond).to(u.radian).value

    # Transformation matrix from Equatorial J2000 to Ecliptic J2000
    TRANSFORM_EQ2EC = np.matrix([
        [1, 0, 0],
        [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
        [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)
    ]])
    
    # Transformation matrix from Ecliptic J2000 to Equatorial J2000
    TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T