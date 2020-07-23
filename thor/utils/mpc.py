import os
import numpy as np
import pandas as pd
from astropy.time import Time

from .io import _readFileLog
from .io import _downloadFile

__all__ = [
    "_unpackMPCDate",
    "_lookupMPC",
    "convertMPCPackedDates",
    "getMPCObsCodeFile",
    "readMPCObsCodeFile",
    "getMPCORBFile",
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

def getMPCObsCodeFile():
    """
    Downloads the JSON-formatted MPC observatory codes file. Checks if a newer version of the file exists online, if so, 
    replaces the previously downloaded file if available. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    directory = os.path.join(os.path.dirname(__file__), "..", "data")
    url = "https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz"
    _downloadFile(directory, url)
    return

def readMPCObsCodeFile(obsCodeFile=None):
    """
    Reads the JSON-formatted MPC observatory codes file. 
    
    Parameters
    ----------
    obsCodeFile : str, optional
        Path to file
        
    Returns
    -------
    observatories : `~pandas.DataFrame`
        DataFrame indexed on observatory code. 
        
    See Also
    --------
    `~thor.utils.mpc.getMPCObsCodeFile` : Downloads the MPC observatory code file.
    """
    if obsCodeFile is None:
        log = _readFileLog(os.path.join(os.path.dirname(__file__), "..", "data", "log.yaml"))
        obsCodeFile = log["obscodes_extended.json.gz"]["location"]
        
    observatories = pd.read_json(obsCodeFile).T
    observatories.rename(columns={
        "Longitude" : "longitude_deg",
        "Name" : "name"},
        inplace=True
    )
    observatories.index.name = 'code'
    return observatories

def getMPCORBFile():
    """
    Downloads the JSON-formatted extended MPC orbit catalog. Checks if a newer version of the file exists online, if so, 
    replaces the previously downloaded file if available. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    directory = os.path.join(os.path.dirname(__file__), "..", "data")
    url = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz" 
    _downloadFile(directory, url)
    return

def readMPCORBFile(mpcOrbFile=None):
    """
    Reads the JSON-formatted extended MPC orbit catalog. 
    
    Parameters
    ----------
    mpcOrbFile : str, optional
        Path to file
    
    Returns
    -------
    mpcorb : `~pandas.DataFrame`
        DataFrame of MPC minor planet orbits.
    
    See Also
    --------
    `~thor.utils.mpc.getMPCORBFile` : Downloads the extended MPC orbit catalog.
    """
    if mpcOrbFile is None:
        log = _readFileLog(os.path.join(os.path.dirname(__file__), "..", "data", "log.yaml"))
        mpcOrbFile = log["mpcorb_extended.json.gz"]["location"]
    
    mpcorb = pd.read_json(mpcOrbFile)
    
    # Add MJDs from JDs
    mpcorb["mjd_tt"] = Time(mpcorb["Epoch"].values, format="jd", scale="tt").tt.mjd
    
    # Re-arange columns
    columns = ['Number', 'Name', 'Principal_desig', 'Other_desigs', 
               'Epoch', 'mjd_tt', 'a', 'e', 'i', 'Node', 'Peri', 'M', 'n', 
               'H', 'G',
               'Tp', 'Orbital_period',
               'Perihelion_dist', 'Aphelion_dist', 'Semilatus_rectum',
               'Synodic_period', 'Orbit_type', 
               'Num_obs', 'Last_obs', 'rms', 'U', 'Arc_years', 'Arc_length', 'Num_opps',  'Perturbers', 'Perturbers_2', 
               'Hex_flags', 'NEO_flag', 'One_km_NEO_flag',
               'PHA_flag', 'Critical_list_numbered_object_flag',
               'One_opposition_object_flag', 'Ref', 'Computer']
    mpcorb = mpcorb[columns]

    # Rename columns, include units where possible
    mpcorb.rename(columns={
        "Number" : "number",
        "Name" :  "name",
        "Principal_desig" : "provisonal_designation",
        "Other_desigs" : "other_provisonal_designations",
        "Epoch" : "jd_tt",
        "a" : "a_au",
        "e" : "e",
        "i" : "i_deg",
        "Node" : "ascNode_deg",
        "Peri" : "argPeri_deg",
        "M" : "meanAnom_deg",
        "n" : "mean_motion_deg_p_day",
        "H" : "H_mag",
        "G" : "G",
        "Tp" : "tPeri_jd",
        "Orbital_period" : "period_yr",
        "Perihelion_dist" : "perihelion_dist_au",
        "Aphelion_dist" : "aphelion_dist_au",
        "Semilatus_rectum" : "p_au",
        "Synodic_period" : "synodic_period_yr",
        "Orbit_type" : "orbit_type",
        "Num_obs" : "num_obs",
        "Last_obs" : "last_obs",
        "rms" : "rms_arcsec",
        "U" : "uncertainty_param",
        "Arc_years" : "arc_yr",
        "Arc_length" : "arc_days",
        "Num_opps" : "num_oppos",
        "Perturbers" : "perturbers1",
        "Perturbers_2" : "perturbers2",
        "Hex_flags" : "hex_flags",
        "NEO_flag" : "neo_flag",
        "One_km_NEO_flag" : "1km_neo_flag",
        "PHA_flag" : "pha_flag",
        "Critical_list_numbered_object_flag" : "critical_list_flag",
        "One_opposition_object_flag" : "1_oppo_flag",
        "Ref" : "reference",
        "Computer" : "computer"
    }, inplace=True)

    return mpcorb