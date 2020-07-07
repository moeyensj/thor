import numpy as np
import pandas as pd
from astropy.time import Time

__all__ = [
    "_unpackMPCDate",
    "_lookupMPC",
    "convertMPCPackedDates",
    "readMPCORBFile"
]


def _unpackMPCDate(epoch_pf):
    # Taken from Lynne Jones' SSO TOOLS. 
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    epoch_pf = str(epoch_pf)
    year = _lookupMPC(epoch_pf[0])*100 + int(epoch_pf[1:3])
    month = _lookupMPC(epoch_pf[3])
    day = _lookupMPC(epoch_pf[4])
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)
    
    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60) 
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)
        
    return isot_string

def _lookupMPC(x):
    # Convert the single character dates into integers.
    try:
        x = int(x)
    except ValueError:
        x = ord(x) - 55
    if x < 0 or x > 31:
        raise ValueError
    return x

def convertMPCPackedDates(pf_tt):
    """
    Convert MPC packed form dates (in the TT time scale) to 
    MJDs in TT. See: https://minorplanetcenter.net/iau/info/PackedDates.html
    for details on the packed date format.

    Parameters
    ----------
    pf_tt : `~numpy.ndarray` (N)
        MPC-style packed form epochs in the TT time scale.

    Returns
    -------
    mjd_tt : `~numpy.ndarray` (N)
        Epochs in TT MJDs.
    """
    pf_tt = np.asarray(pf_tt)
    isot_tt = np.empty(len(pf_tt), dtype="<U32")
    
    for i, epoch in enumerate(pf_tt):
        isot_tt[i] = _unpackMPCDate(epoch)
    
    epoch = Time(isot_tt, format="isot", scale="tt")
    return epoch.tt.mjd

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
              "lastObsInOrbitSolution" : str}
    
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
    
    # Drop population line breaks
    mpcorb.dropna(inplace=True)
    mpcorb.reset_index(inplace=True, drop=True)
    
    # Convert packed form dates into something interpretable
    mjd_tt = convertMPCPackedDates(mpcorb["epoch_pf_TT"].values)
    mpcorb["mjd_tt"] = mjd_tt
    
    # Convert dtypes (misread due to NaN values)
    mpcorb["numObs"] = mpcorb["numObs"].astype(int)
    mpcorb["numOppos"] = mpcorb["numOppos"].astype(int)
    mpcorb["lastObsInOrbitSolution"] = mpcorb["lastObsInOrbitSolution"].astype(int).astype(str)
    
    # Organize columns
    mpcorb = mpcorb[[
        "designation", 
        "H", 
        "G",
        "epoch_pf_TT", 
        "mjd_tt",
        "a_au",
        "e",
        "i_deg",
        "ascNode_deg",
        "argPeri_deg",
        "meanAnom_deg", 
        "n_deg_p_day",
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
        "lastObsInOrbitSolution", 
    ]]
    
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