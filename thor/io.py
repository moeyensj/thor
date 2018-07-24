import numpy as np
import pandas as pd
import sqlite3 as sql

__all__ = ["readMPCORBFile", 
           "readORBFile",
           "readEPHFile", 
           "buildObjectDatabase",
           "calcNight"]

def readMPCORBFile(file,
                   con=None):
    """
    Read MPCORB.DAT file into a pandas DataFrame.

    For more details about the MPCORB file:
    https://www.minorplanetcenter.net/iau/MPCORB.html
    
    Parameters
    ----------
    file : str
        Path to MPCORB.dat
    con : `~sqlite3.Connection`, optional
        If a database connection is passed, will save
        DataFrame into database as mpcOrbitCat table.
        
    Returns
    -------
    `~pandas.DataFrame` or None
        If database connection is not passed, will
        return DataFrame of the MPC Orbit Catalog file.
    """
    columns = ["designation",
               "H",
               "G", 
               "epoch_pf_TT",
               "meanAnom_deg",
               "argPeri_deg",
               "ascNode_deg",
               "i_deg",
               "e", 
               "n_deg_p_day",
               "a_au",
               "U",
               "ref",
               "numObs",
               "numOppos",
               "obsArc",
               "rmsResid_arcsec",
               "coarsePerturbers",
               "precisePerturbers",
               "compName", 
               "flags",
               "readableDesignation",
               "lastObsInOrbitSolution"]

    # See: https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html
    column_spec = [(0, 7),
                   (8, 13),
                   (14, 19),
                   (20, 25),
                   (26, 35),
                   (37, 46),
                   (48, 57),
                   (59, 68),
                   (70, 79),
                   (80, 91),
                   (92, 103),
                   (105, 106),
                   (107, 116),
                   (117, 122),
                   (123, 126),
                   (127, 136),
                   (137, 141),
                   (142, 145),
                   (146, 149),
                   (150, 160),
                   (161, 165),
                   (166, 194),
                   (194, 202)]

    dtypes = {"H" : np.float64,
              "G" : np.float64,
              "epoch_pf_TT" : str,
              "meanAnom_deg" : np.float64,
              "argPeri_deg" : np.float64,
              "ascNode_deg" : np.float64,
              "i_deg" : np.float64,
              "e" : np.float64,
              "n_deg_p_day" : np.float64,
              "a_au" : np.float64,
              "U" : str,
              "ref" : str,
              "numObs" : np.int64,
              "numOppos" : np.int64,
              "obsArc" : str,
              "rmsResid_arcsec" : np.float64,
              "coarsePerturbers" : str,
              "precisePerturbers" : str,
              "compName" : str,
              "lastObsInOrbitSolution" : np.int64}
    
    converters = {"designation": lambda x: str(x),
                  "readableDesignation" : lambda x: str(x),
                  "flags" : lambda x: str(x)}
    
    mpcorb = pd.read_fwf(file,
                         skiprows=43,
                         colspecs=column_spec,
                         header=None,
                         index_col=False, 
                         names=columns,
                         dtypes=dtypes,
                         converters=converters)
    
    if con is not None:
        print("Reading MPCORB file to database...")
        mpcorb.to_sql("mpcOrbitCat", con, index=False, if_exists="append")
        print("Creating index on object names...")
        con.execute("CREATE INDEX designation_mpcorb ON mpcOrbitCat (designation)")
        con.commit()
        print("Done.")
        print("")
    else: 
        return mpcorb

def readORBFile(file,
                elementType="keplerian",
                con=None):
    """
    Read an oorb .orb file into a pandas DataFrame.
    
    Parameters
    ----------
    file : str
        Path to file.orb
    elementType : str, optional
        Orbital element type of input .orb file. Should be consistent 
        with default defined in `~thor.Config.oorbConfig`.
        [Default = 'keplerian']
    con : `~sqlite3.Connection`, optional
        If a database connection is passed, will save
        DataFrame into database as oorbOrbitCat table.
        [Default = None]
        
    Returns
    -------
    `~pandas.DataFrame` or None
        If database connection is not passed, will
        return DataFrame of the Oorb orbit file.   
    """
    if elementType == "keplerian":
        columns = ["designation",
                   "a_au",
                   "e",
                   "i_deg",
                   "ascNode_deg",
                   "argPeri_deg",
                   "meanAnom_deg", 
                   "epoch_TT_mjd",
                   "H",
                   "G"]
        
    elif elementType == "cartesian":
        columns = ["designation",
                   "x_ec_au",
                   "y_ec_au",
                   "z_ec_au",
                   "dx_ec/dt_au_p_day",
                   "dy_ec/dt_au_p_day",
                   "dz_ec/dt_au_p_day", 
                   "epoch_TT_mjd",
                   "H",
                   "G"]
    else:
        raise ValueError("elementType should be one of 'keplerian' or 'cartesian'")
    
    # See: https://github.com/oorb/oorb/blob/master/main/io.f90#L3477
    # See: https://github.com/oorb/oorb/blob/master/main/io.f90#L3652
    # See: https://github.com/oorb/oorb/blob/master/main/io.f90#L3661
    # (A16,6(1X,E21.14),1X,F16.8,1X,F9.5,1X,F9.6)
    column_spec = [(0, 16),
                   (17, 38),
                   (39, 60),
                   (61, 82),
                   (83, 104),
                   (105, 126),
                   (127, 148),
                   (149, 165),
                   (166, 175),
                   (176, 186)]

    dtypes = {name : np.float64 for name in columns[1:]} 
    
    converters = {"designation": lambda x: str(x)}
            
    orb = pd.read_fwf(file,
                      skiprows=4,
                      colspecs=column_spec,
                      header=None, 
                      index_col=False,
                      names=columns,
                      dtypes=dtypes,
                      converters=converters)
    
    if con is not None:
        print("Reading oorb orbit file to database...")
        orb.to_sql("oorbOrbitCat", con, index=False, if_exists="append")
        print("Creating index on object names...")
        con.execute("CREATE INDEX designation_oorborb ON oorbOrbitCat (designation)")
        con.commit()
        print("Done.")
        print("")
    else:
        return orb

def readEPHFile(file,
                con=None,
                chunksize=100000):
    """
    Read an oorb .eph file into a pandas DataFrame.
    
    Parameters
    ----------
    file : str
        Path to file.eph
    con : `~sqlite3.Connection`, optional
        If a database connection is passed, will save
        DataFrame into database as ephemeris table.
    chunksize : int, optional
        Read file (and save to database) in chunks of this
        size.
        
    Returns
    -------
    `~pandas.DataFrame`
        Oorb ephemeris file as a DataFrame.
        
    """
    columns = ["designation",
               "code",
               "mjd_utc",
               "Delta_au",
               "RA_deg",
               "Dec_deg",
               "dDelta/dt_au_p_day",
               "dRA/dt_deg_p_day",
               "dDec/dt_deg_p_day",
               "VMag",
               "Alt_deg",
               "PhaseAngle_deg",
               "LunarElon_deg",
               "LunarAlt_deg",
               "LunarPhase",
               "SolarElon_deg",
               "SolarAlt_deg",
               "r_au",
               "HLon_deg",
               "HLat_deg",
               "TLon_deg",
               "TLat_deg",
               "TOCLon_deg",
               "TOCLat_deg",
               "HOCLon_deg",
               "HOCLat_deg",
               "TOppLon_deg",
               "TOppLat_deg",
               "HEclObj_X_au",
               "HEclObj_Y_au",
               "HEclObj_Z_au",
               "HEclObj_dX/dt_au_p_day",
               "HEclObj_dY/dt_au_p_day",
               "HEclObj_dZ/dt_au_p_day",
               "HEclObsy_X_au",
               "HEclObsy_Y_au",
               "HEclObsy_Z_au",
               "EccAnom",
               "TrueAnom",
               "PosAngle_deg"]
    
    # See: https://github.com/oorb/oorb/blob/master/main/oorb.f90#L7089
    # (A,A11,1X,A,1X,38(A18,1X))
    column_spec = [(0, 11),
                   (12, 21),
                   (22, 40),
                   (41, 59),
                   (60, 78),
                   (79, 97),
                   (98, 116),
                   (117, 135),
                   (136, 154),
                   (155, 173),
                   (174, 192),
                   (193, 211),
                   (212, 230),
                   (231, 249),
                   (250, 268),
                   (269, 287),
                   (288, 306),
                   (307, 325),
                   (326, 344),
                   (345, 363),
                   (364, 382),
                   (383, 401),
                   (402, 420),
                   (421, 439),
                   (440, 458),
                   (459, 477),
                   (478, 496),
                   (497, 515),
                   (516, 534),
                   (535, 553),
                   (554, 572),
                   (573, 591),
                   (592, 610),
                   (611, 629),
                   (630, 648),
                   (649, 667),
                   (668, 686),
                   (687, 705),
                   (706, 724),
                   (725, 743)]
    
    dtypes = {name : np.float64 for name in columns[2:]} 
              
    converters = {"designation": lambda x: str(x)}
    
    if con is not None:
        print("Reading oorb ephemeris file to database...")
        for chunk in pd.read_fwf(file, 
                                 skiprows=1,
                                 colspecs=column_spec,
                                 header=None, 
                                 index_col=False,
                                 names=columns,
                                 dtypes=dtypes,
                                 converters=converters,
                                 chunksize=chunksize):
        
            chunk["night"] = calcNight(chunk["mjd_utc"].values)
            
            chunk.to_sql("ephemeris",
                         con,
                         index=True,
                         index_label="obsId",
                         if_exists="append",
                         chunksize=chunksize)
            
        print("Creating index on object names...")
        con.execute("CREATE INDEX designation_oorbeph ON ephemeris (designation)")
        print("Creating index on observation ids...")
        con.execute("CREATE INDEX obsId_oorbeph ON ephemeris (obsId)")
        print("Creating index on nights...")
        con.execute("CREATE INDEX night_oorbeph ON ephemeris (night)")
        print("Creating positional indexes...")
        con.execute("CREATE INDEX ra_oorbeph ON ephemeris (RA_deg)")
        con.execute("CREATE INDEX dec_oorbeph ON ephemeris (Dec_deg)")
        con.execute("CREATE INDEX lon_oorbeph ON ephemeris (HLon_deg)")
        con.execute("CREATE INDEX lat_oorbeph ON ephemeris (HLat_deg)")
        con.execute("CREATE INDEX r_oorbeph ON ephemeris (r_au)")
        con.execute("CREATE INDEX delta_oorbeph ON ephemeris (Delta_au)")
        con.commit()
        print("Done.")
        
    else:
        eph = pd.read_fwf(file, 
                          skiprows=1, 
                          colspecs=column_spec,
                          header=None, 
                          index_col=False,
                          names=columns,
                          dtypes=dtypes,
                          converters=converters)
        return eph


def buildObjectDatabase(database,
                        mpcorbFile=None,
                        orbFile=None,
                        ephFile=None,
                        elementType="keplerian",
                        chunksize=100000):
    """
    Prepare object database and populate with choice of MPCORB catalogue,
    oorb orbit catalogue and/or oorb ephemeris. 
    
    Parameters
    ----------
    database : str
        Path to database.
    mpcorbFile : str, optional
        Path to MPCORB file. Will be read into table.
    orbFile : str, optional
        Path to oorb orbit file. Will be read into table.
    ephFile : str, optional
        Path to oorb ephemeris file. Will be read into table.
    elementType : str, optional
        Orbital element type of input .orb file. Should be consistent 
        with default defined in `~thor.Config.oorbConfig`.
        [Default = 'keplerian']
    chunksize : int, optional
        Number of lines per chunk to break ephemeris file into when reading
        into database.
        [Default = 100000]
    
    Returns
    -------
    `~sqlite3.connection`
        Connection to database.
    """
    
    con = sql.connect(database)
    
    if mpcorbFile is not None:
        print("Building mpcOrbitCat table...")
        con.execute("""
            CREATE TABLE mpcOrbitCat (
                "designation" VARCHAR,
                "H" REAL,
                "G" REAL, 
                "epoch_pf_TT" VARCHAR,
                "meanAnom_deg" REAL,
                "argPeri_deg" REAL,
                "ascNode_deg" REAL,
                "i_deg" REAL,
                "e" REAL, 
                "n_deg_p_day" REAL,
                "a_au" REAL,
                "U" VARCHAR,
                "ref" VARCHAR,
                "numObs" INTEGER,
                "numOppos" INTEGER,
                "obsArc" VARCHAR,
                "rmsResid_arcsec" REAL,
                "coarsePerturbers" VARCHAR,
                "precisePerturbers" VARCHAR,
                "compName" VARCHAR,
                "flags" VARCHAR,
                "readableDesignation" VARCHAR,
                "lastObsInOrbitSolution" INTEGER
            );""")
        mpcorb = readMPCORBFile(mpcorbFile, con=con)
        
    if orbFile is not None:
        print("Building oorbOrbitCat table...")
        if elementType == "keplerian":
            con.execute("""
            CREATE TABLE oorbOrbitCat (
                "designation" VARCHAR,
                "a_au" REAL,
                "e" REAL,
                "i_deg" REAL,
                "ascNode_deg" REAL,
                "argPeri_deg" REAL,
                "meanAnom_deg" REAL, 
                "epoch_TT_mjd" VARCHAR,
                "H" REAL,
                "G" REAL
            );""")
        elif elementType == "cartesian":
            con.execute("""
            CREATE TABLE oorbOrbitCat (
                "designation" VARCHAR,
                "x_ec_au" REAL,
                "y_ec_au" REAL,
                "z_ec_au" REAL,
                "dx_ec/dt_au_p_day" REAL,
                "dy_ec/dt_au_p_day" REAL,
                "dz_ec/dt_au_p_day" REAL, 
                "epoch_TT_mjd" VARCHAR,
                "H" REAL,
                "G" REAL
            );""")
        else:
            raise ValueError("elementType should be one of 'keplerian' or 'cartesian'")

        orb = readORBFile(orbFile, con=con)
        
    if ephFile is not None:
        print("Building ephemeris table...")
        con.execute("""
            CREATE TABLE ephemeris (
                "obsId" INTEGER NOT NULL PRIMARY KEY,
                "designation" VARCHAR,
                "code" VARCHAR,
                "mjd_utc" REAL,
                "night" INTEGER,
                "Delta_au" REAL,
                "RA_deg" REAL,
                "Dec_deg" REAL,
                "dDelta/dt_au_p_day" REAL,
                "dRA/dt_deg_p_day" REAL,
                "dDec/dt_deg_p_day" REAL,
                "VMag" REAL,
                "Alt_deg" REAL,
                "PhaseAngle_deg" REAL,
                "LunarElon_deg" REAL,
                "LunarAlt_deg" REAL,
                "LunarPhase" REAL,
                "SolarElon_deg" REAL,
                "SolarAlt_deg" REAL,
                "r_au" REAL,
                "HLon_deg" REAL,
                "HLat_deg" REAL,
                "TLon_deg" REAL,
                "TLat_deg" REAL,
                "TOCLon_deg" REAL,
                "TOCLat_deg" REAL,
                "HOCLon_deg" REAL,
                "HOCLat_deg" REAL, 
                "TOppLon_deg" REAL,
                "TOppLat_deg" REAL,
                "HEclObj_X_au" REAL,
                "HEclObj_Y_au" REAL,
                "HEclObj_Z_au" REAL,
                "HEclObj_dX/dt_au_p_day" REAL,
                "HEclObj_dY/dt_au_p_day" REAL,
                "HEclObj_dZ/dt_au_p_day" REAL,
                "HEclObsy_X_au" REAL,
                "HEclObsy_Y_au" REAL,
                "HEclObsy_Z_au" REAL,
                "EccAnom" REAL,
                "TrueAnom" REAL,
                "PosAngle_deg" REAL
            );
        """)
        eph = readEPHFile(ephFile, con=con, chunksize=chunksize)

    return con

def calcNight(mjd, midnight=0.166):
    """
    Calculate the integer night for any MJD.

    Parameters
    ----------
    mjd : float or `~numpy.ndarray`
        MJD to convert.
    midNight : float, optional
        Midnight in MJD at telescope site.

    Returns
    -------
    int or `~numpy.ndarray`
        Night of observation
    """
    night = mjd + 0.5 - midnight
    return night.astype(int)
    