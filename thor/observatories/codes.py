import io
import os
import urllib
import filecmp

__all__ = [
    "getMPCObsCodeFile"
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
    obscodes_old = os.path.join(os.path.dirname(__file__), "data/ObsCodes_old.html")
    if os.path.isfile(obscodes) is not True:
        print("Could not find MPC Observatory Code file. Attempting download...")
        url = 'https://www.minorplanetcenter.net/iau/lists/ObsCodes.html'  
        response = urllib.request.urlopen(url)  
        print("File succesfully downloaded.")
        
        f = io.BytesIO(response.read())
        with open(obscodes, 'wb') as outfile:
            outfile.write(f.read())
        print("Done.")
    else:
        print("MPC Observatory Codes file exists, checking if there is a new version...")
        obscodes_new = os.path.join(os.path.dirname(__file__), "data/.ObsCodes.html")
        url = 'https://www.minorplanetcenter.net/iau/lists/ObsCodes.html'  
        response = urllib.request.urlopen(url)  
        
        f = io.BytesIO(response.read())
        with open(obscodes_new, 'wb') as outfile:
            outfile.write(f.read())
        
        # Check if newly downloaded file is different compared to the previously downloaded one
        # If so, replace the old version with the newer one and save the old version (in case the new one 
        # is broken)
        if not filecmp.cmp(obscodes, obscodes_new):
            print("File has been updated: replacing old file with newer version.")
            print("Saving older file to ObsCodes_old.html")
            os.rename(obscodes, obscodes_old)
            os.rename(obscodes_new, obscodes)
        else:
            os.remove(obscodes_new)
            print("Latest version already downloaded.")

    return