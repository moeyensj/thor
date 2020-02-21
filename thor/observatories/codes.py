import os
import pandas as pd

from ..utils import _downloadFile

__all__ = [
    "getMPCObsCodeFile",
    "readMPCObsCodeFile"
]

def getMPCObsCodeFile():
    """
    Downloads the MPC Observatory Codes file. Checks if a newer version of the file exists online, if so, 
    replaces the previously downloaded file if available. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    obscodes = os.path.join(os.path.dirname(__file__), "data/ObsCodes.html")
    url = 'https://www.minorplanetcenter.net/iau/lists/ObsCodes.html'  
    _downloadFile(obscodes, url, update=True)
    return

def readMPCObsCodeFile(obsCodeFile=None):
    """
    Reads the MPC Observatory Code file. 
    
    Parameters
    ----------
    obsCodeFile : str, optional
        Path to file
        
    Returns
    -------
    observatories : `~pandas.DataFrame`
        DataFrame indexed on observatory code. 
    """
    if obsCodeFile is None:
        obsCodeFile = os.path.join(os.path.dirname(__file__), "data/ObsCodes.html")
    observatories = pd.read_fwf(obsCodeFile, 
                                colspecs=[
                                    (0, 4),
                                    (4, 13),
                                    (13, 21),
                                    (21, 30),
                                    (30, 250)],
                                header=1, 
                                skipfooter=1)
    observatories.rename(columns={"Code" : "code", "Long." : "longitude_deg", "Name" : "name"}, inplace=True)
    observatories = observatories.set_index("code")
    return observatories

