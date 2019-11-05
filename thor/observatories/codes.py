import os

from ..utils import _downloadFile

__all__ = [
    "getMPCObsCodeFile",
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

