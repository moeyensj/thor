from astroquery.jplhorizons import Horizons
from .astropy import _checkTime

__all__ = [
    "getHorizonsVectors",
    "getHorizonsElements",
    "getHorizonsEphemeris"
]

def getHorizonsVectors(obj_id, times, location="@sun", id_type="smallbody", aberrations="geometric"):
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
    id_type : {'majorbody', 'smallbody', 'designation', 
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
    aberrations : {'geometric', 'astrometric', 'apparent'}
        Adjust state for one of three different aberrations. 
        
    Returns
    -------
    vectors : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time 
        of the object at each time. 
    """
    _checkTime(times, "times")
    obj = Horizons(
        id=obj_id, 
        epochs=times.tdb.mjd,
        location=location,
        id_type=id_type,
    )
    vectors = obj.vectors(
        refplane="ecliptic",
        aberrations=aberrations,
    ).to_pandas()
    return vectors

def getHorizonsElements(obj_id, times, location="@sun", id_type="smallbody"):
    """
    Query JPL Horizons (through astroquery) for an object's
    elements at the given times.
    
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
    id_type : {'majorbody', 'smallbody', 'designation', 
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
        
    Returns
    -------
    elements : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time 
        of the object at each time. 
    """
    _checkTime(times, "times")
    obj = Horizons(
        id=obj_id, 
        epochs=times.tdb.mjd,
        location=location,
        id_type=id_type,
    )
    elements = obj.elements(
        refsystem="J2000",
        refplane="ecliptic",
        tp_type="absolute"
    ).to_pandas()
    return elements

def getHorizonsEphemeris(obj_id, times, location, id_type="smallbody"):
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
    id_type : {'majorbody', 'smallbody', 'designation', 
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.
    
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
        id_type=id_type
    )
    eph = obj.ephemerides(
        # RA, DEC, r, r_rate, delta, delta_rate, lighttime
        quantities="1, 2, 19, 20, 21",
        extra_precision=True
    ).to_pandas()
    return eph