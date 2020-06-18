import numpy as np

class Constants:

    # km to au
    KM_P_AU = 149597870.700

    # seconds to days
    S_P_DAY = 86400.0
    
    # Speed of light: AU per day (173.14463267424034) (299792.458 km/s -- DE431/DE430)
    C = 299792.458 / KM_P_AU * S_P_DAY
    
    # Gravitational constant:  AU**3 / M_sun / d**2 (0.295912208285591100E-3 -- DE431/DE430)
    G = 0.295912208285591100E-3
    
    # Solar Mass: M_sun (1.0)
    M_SUN = 1.0
    
    # Earth Equatorial Radius: km (6378.1363 km -- DE431/DE430)
    R_EARTH = 6378.1363 / KM_P_AU

    # Mean Obliquity at J2000: radians (84381.448 arcseconds -- DE431/DE430)
    OBLIQUITY = 84381.448 * np.pi / (180.0 * 3600.0)

    # Transformation matrix from Equatorial J2000 to Ecliptic J2000
    TRANSFORM_EQ2EC = np.array([
        [1, 0, 0],
        [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)],
        [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)
    ]])
    
    # Transformation matrix from Ecliptic J2000 to Equatorial J2000
    TRANSFORM_EC2EQ = TRANSFORM_EQ2EC.T