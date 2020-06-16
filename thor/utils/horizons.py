from astroquery.jplhorizons import Horizons
from .astropy import _checkTime

__all__ = [
    "getHorizonsVectors",
    "getHorizonsEphemeris"
]

def getHorizonsVectors(obj_id, times, location="@sun"):
    """
    Query JPL Horizons (through astroquery) for an object's
    state vectors at the given times.
    
    Parameters
    ----------
    obj_id : str
        Object ID / designation recognizable by HORIZONS. 
    times : `~astropy.core.time.Time`
        Astropy time object at which to gather state vectors.
    location : str, optional
    	Location of the origin typically a NAIF code.
    	('0' or '@ssb' for solar system barycenter, '10' or '@sun' for heliocenter)
    	[Default = '@sun']
    	
    Returns
    -------
    vectors : `~pandas.DataFrame`
        Dataframe containing the RA, DEC, r, r_rate, delta, delta_rate and light time 
        of the object at each time. 
    """
    _checkTime(times, "times")
    obj = Horizons(
        id=obj_id, 
        epochs=times.tdb.mjd,
        location=location,
        id_type="smallbody",
    )
    vectors = obj.vectors(
        refplane="ecliptic",
        aberrations="geometric",
    ).to_pandas()
    return vectors

def getHorizonsEphemeris(obj_id, times, location):
    """
    Query JPL Horizons (through astroquery) for an object's
    ephemerides at the given times viewed from the given location.
    
    Parameters
    ----------
    obj_id : str
        Object ID / designation recognizable by HORIZONS. 
    times : `~astropy.core.time.Time`
        Astropy time object at which to gather ephemerides.
    location : str
        Location of the observer which can be an MPC observatory code (eg, 'I41', 'I11')
        or a NAIF code ('0' for solar system barycenter, '10' for heliocenter)
    
    Returns
    -------
    eph : `~pandas.DataFrame`
        Dataframe containing the RA, DEC, r, r_rate, delta, delta_rate and light time 
        of the object at each time. 
    """
    _checkTime(times, "times")
    obj = Horizons(
        id=obj_id, 
        epochs=times.utc.mjd, 
        location=location,
        id_type="smallbody"
    )
    eph = obj.ephemerides(
        # RA, DEC, r, r_rate, delta, delta_rate, lighttime
        quantities="1, 19, 20, 21",
        extra_precision=True
    ).to_pandas()
    return eph