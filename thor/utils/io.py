import os
import io
import shutil
import urllib
import filecmp

__all__ = [
    "_downloadFile"
]


def _downloadFile(fname, url, update=False):
    """
    Downloads a file from the url and saves it to fname. If update is True, will check if the currently downloaded version
    is the latest version. 
    
    """
    # The file doesn't exist yet so try to download it 
    if os.path.isfile(fname) is not True:
        response = urllib.request.urlopen(url)  
        print("File not found. Attempting download...")
        
        f = io.BytesIO(response.read())
        with open(fname, 'wb') as outfile:
            outfile.write(f.read())
        print("Done.")
    
    # The file does exist, if update is true download it again
    else:
        print("File has already been downloaded.")
        if update:
            print("Checking if updated file exists...")
            fname_new = os.path.join(os.path.dirname(fname), ".{}".format(os.path.basename(fname)))
            fname_old = os.path.join(os.path.dirname(fname), "{}_old".format(os.path.basename(fname)))
            response = urllib.request.urlopen(url)  
        
            f_new = io.BytesIO(response.read())
            with open(fname_new, 'wb') as outfile:
                outfile.write(f_new.read())
        
            # Check if newly downloaded file is different compared to the previously downloaded one
            # If so, replace the old version with the newer one and save the old version (in case the new one 
            # is broken)
            if not filecmp.cmp(fname, fname_new):
                print("File outdated! Replacing with latest version.")
                os.rename(fname, fname_old)
                os.rename(fname_new, fname)
            else:
                print("Latest version of the file already acquired.")
