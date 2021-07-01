import pandas as pd
from astroquery.jplhorizons import Horizons
from .astropy import _checkTime

__all__ = [
    "getHorizonsVectors",
    "getHorizonsElements",
    "getHorizonsEphemeris",
    "getHorizonsObserverState"
]

def getHorizonsVectors(
        obj_ids,
        times,
        location="@sun",
        id_type="smallbody",
        aberrations="geometric",
    ):
    """
    Query JPL Horizons (through astroquery) for an object's
    state vectors at the given times.

    Parameters
    ----------
    obj_ids : `~numpy.ndarray` (N)
        Object IDs / designations recognizable by HORIZONS.
    times : `~astropy.core.time.Time` (M)
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
    dfs = []
    for obj_id in obj_ids:
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        vectors = obj.vectors(
            refplane="ecliptic",
            aberrations=aberrations,
            cache=False
        ).to_pandas()
        dfs.append(vectors)

    vectors = pd.concat(
        dfs,
        ignore_index=True
    )
    return vectors

def getHorizonsElements(
        obj_ids,
        times,
        location="@sun",
        id_type="smallbody"
    ):
    """
    Query JPL Horizons (through astroquery) for an object's
    elements at the given times.

    Parameters
    ----------
    obj_ids : `~numpy.ndarray` (N)
        Object IDs / designations recognizable by HORIZONS.
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
    dfs = []
    for obj_id in obj_ids:
        obj = Horizons(
            id=obj_id,
            epochs=times.tdb.mjd,
            location=location,
            id_type=id_type,
        )
        elements = obj.elements(
            refsystem="J2000",
            refplane="ecliptic",
            tp_type="absolute",
            cache=False
        ).to_pandas()
        dfs.append(elements)

    elements = pd.concat(
        dfs,
        ignore_index=True
    )
    return elements

def getHorizonsEphemeris(
        obj_ids,
        observers,
        id_type="smallbody"
    ):
    """
    Query JPL Horizons (through astroquery) for an object's
    ephemerides at the given times viewed from the given location.

    Parameters
    ----------
    obj_ids : `~numpy.ndarray` (N)
        Object IDs / designations recognizable by HORIZONS.
    observers : dict or `~pandas.DataFrame`
        A dictionary with observatory/location codes as keys and observation_times (`~astropy.time.core.Time`) as values.
        Location of the observer which can be an MPC observatory code (eg, 'I41', 'I11')
        or a NAIF code ('0' for solar system barycenter, '10' for heliocenter)
    id_type : {'majorbody', 'smallbody', 'designation',
               'name', 'asteroid_name', 'comet_name', 'id'}
        ID type, Horizons will find closest match under any given type.

    Returns
    -------
    ephemeris : `~pandas.DataFrame`
        Dataframe containing the RA, DEC, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    dfs = []
    for orbit_id, obj_id in enumerate(obj_ids):
        for observatory_code, observation_times in observers.items():
            _checkTime(observation_times, "observation_times")
            obj = Horizons(
                id=obj_id,
                epochs=observation_times.utc.mjd,
                location=observatory_code,
                id_type=id_type
            )
            ephemeris = obj.ephemerides(
                # RA, DEC, r, r_rate, delta, delta_rate, lighttime
                #quantities="1, 2, 19, 20, 21",
                extra_precision=True
            ).to_pandas()
            ephemeris["orbit_id"] = [orbit_id for i in range(len(ephemeris))]
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
            ephemeris["mjd_utc"] = observation_times.utc.mjd

            dfs.append(ephemeris)

    ephemeris = pd.concat(dfs)
    ephemeris.sort_values(
        by=["orbit_id", "observatory_code", "datetime_jd"],
        inplace=True,
        ignore_index=True
    )
    return ephemeris

def getHorizonsObserverState(
        observatory_codes,
        observation_times,
        origin="heliocenter",
        aberrations="geometric"
    ):
    """
    Query JPL Horizons (through astroquery) for an object's
    elements at the given times.

    Parameters
    ----------
    observatory_codes : list or `~numpy.ndarray`
        MPC observatory codes.
    observation_times : `~astropy.time.core.Time`
        Epochs for which to find the observatory locations.
    origin : {'barycenter', 'heliocenter'}
        Return observer state with heliocentric or barycentric origin.
    aberrations : {'geometric', 'astrometric', 'apparent'}
        Adjust state for one of three different aberrations.

    Returns
    -------
    vectors : `~pandas.DataFrame`
        Dataframe containing the full cartesian state, r, r_rate, delta, delta_rate and light time
        of the object at each time.
    """
    _checkTime(observation_times, "observation_times")

    if origin == "heliocenter":
        origin_horizons = "sun"
    elif origin == "barycenter":
        origin_horizons = "ssb"
    else:
        err = (
            "origin should be one of {'heliocenter', 'barycenter'}"
        )
        raise ValueError(err)

    dfs = []
    for code in observatory_codes:
        obj = Horizons(
            id=origin_horizons,
            epochs=observation_times.tdb.mjd,
            location=code,
            id_type="majorbody",
        )
        vectors = obj.vectors(
            refplane="ecliptic",
            aberrations=aberrations,
            cache=False,
        ).to_pandas()

        vectors = vectors.drop(columns="targetname")
        vectors.insert(0, "observatory_code", [code for i in range(len(vectors))])
        vectors.loc[:, ["x", "y", "z", "vx", "vy", "vz"]] *= -1
        dfs.append(vectors)

    vectors = pd.concat(
        dfs,
        ignore_index=True
    )
    return vectors
