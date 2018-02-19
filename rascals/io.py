import numpy as np
import pandas as pd
import sqlite3 as sql

__all__ = ["readMPCORBFile", "readORBFile",
           "readEPHFile", "buildObjectDatabase",
           "calcNight"]

def readMPCORBFile(file,
                   con=None):
    """
    Read MPCORB.DAT file into a pandas DataFrame.
    
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
               "M0_deg",
               "argPeri_deg",
               "Omega_deg",
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
               "compName", "flags1",
               "flags2",
               "readableDesignation",
               "lastObsInOrbitSolution"]
    
    dtypes = {"H" : np.float64,
              "G" : np.float64,
              "epoch_pf_TT" : str,
              "M0_deg" : np.float64,
              "argPeri_deg" : np.float64,
              "Omega_deg" : np.float64,
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
                  "flags1" : lambda x: str(x),
                  "flags2" : lambda x: str(x)}
    
    mpcorb = pd.read_fwf(file,
                         skiprows=43,
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
                con=None):
    """
    Read an oorb .orb file into a pandas DataFrame.
    
    Parameters
    ----------
    file : str
        Path to file.orb
    con : `~sqlite3.Connection`, optional
        If a database connection is passed, will save
        DataFrame into database as oorbOrbitCat table.
        
    Returns
    -------
    `~pandas.DataFrame` or None
        If database connection is not passed, will
        return DataFrame of the Oorb orbit file.   
    """
    columns = ["designation",
               "x_ec_au",
               "y_ec_au",
               "z_ec_au",
               "dx_ec/dt_au_p_day",
               "dy_ec/dt_au_p_day",
               "dz_ec/dt_au_p_day", 
               "epoch_TT",
               "H",
               "G"]

    dtypes = {name : np.float64 for name in columns[1:]} 
    
    converters = {"designation": lambda x: str(x)}
            
    orb = pd.read_fwf(file,
                      skiprows=4,
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
    
    dtypes = {name : np.float64 for name in columns[2:]} 
              
    converters = {"designation": lambda x: str(x)}
    
    if con is not None:
        print("Reading oorb ephemeris file to database...")
        for chunk in pd.read_fwf(file, 
                                 skiprows=1,
                                 index_col=False, 
                                 chunksize=chunksize,
                                 names=columns,
                                 dtypes=dtypes,
                                 converters=converters):
        
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
                          index_col=False, 
                          names=columns,
                          dtypes=dtypes,
                          converters=converters)
        return eph

def buildObjectDatabase(database,
                        mpcorbFile=None,
                        orbFile=None,
                        ephFile=None,
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
                "M0_deg" REAL,
                "argPeri_deg" REAL,
                "Omega_deg" REAL,
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
                "flags1" VARCHAR,
                "flags2" VARCHAR,
                "readableDesignation" VARCHAR,
                "lastObsInOrbitSolution" INTEGER
            );""")
        mpcorb = readMPCORBFile(mpcorbFile, con=con)
        
    if orbFile is not None:
        print("Building oorbOrbitCat table...")
        con.execute("""
            CREATE TABLE oorbOrbitCat (
                "designation" VARCHAR,
                "x_ec_au" REAL,
                "y_ec_au" REAL,
                "z_ec_au" REAL,
                "dx_ec/dt_au_p_day" REAL,
                "dy_ec/dt_au_p_day" REAL,
                "dz_ec/dt_au_p_day" REAL, 
                "epoch_TT" VARCHAR,
                "H" REAL,
                "G" REAL
            );""")
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
    