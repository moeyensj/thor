import numpy as np
from astropy import units as u
from astropy.constants import codata2014 as c

AU_TO_KM = 6.684587122268446e-09

class Constants:
    
    # Speed of light: AU per day (173.14463267424034)
    C = c.c.to(u.au / u.d).value 
    
    # Gravitational constant:  AU**3 / M_sun / d**2 (0.295912208285591100E-3 -- DE431/DE430)
    G = 0.295912208285591100E-3
    
    # Solar Mass: M_sun (1.0)
    M_SUN = 1.0
    
    # Earth Mass: M_sun (3.0034893488507934e-06)
    M_EARTH = u.M_earth.to(u.M_sun)
    
    # Earth Equatorial Radius: km (6378.1363 -- DE431/DE430)
    R_EARTH = (6378.1363 * u.km).to(u.AU).value

    # Mean Obliquity at J2000: radians (0.40909280422232897)
    OBLIQUITY = (84381.448 * u.arcsecond).to(u.radian).value

    # km to au
    KM_TO_AU = u.km.to(u.AU)

    # seconds to days
    S_TO_DAY = u.s.to(u.d)

    # Transformation matrix from Equatorial J2000 to Ecliptic J2000
    TRANSFORM_EQ2EC = np.array([
        [1, 0, 0],
        [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
        [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)
    ]])
    
    # Transformation matrix from Ecliptic J2000 to Equatorial J2000
    TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T