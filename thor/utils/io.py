import os
import io
import shutil
import urllib
import filecmp

__all__ = [
    "_downloadFile"
]


def _downloadFile(file_name, url, update=False):
    """
    Downloads a file from the url and saves it to file_name. If update is True, will check if the currently downloaded version
    is the latest version. 
    
    Parameters
    ----------
    file_name : str
        Path to file.
    url : str
        Location of file online.
    
    """
    # The file doesn't exist yet so try to download it 
    if os.path.isfile(file_name) is not True:
        response = urllib.request.urlopen(url)  
        print("File not found. Attempting download...")
        
        f = io.BytesIO(response.read())
        with open(file_name, 'wb') as outfile:
            outfile.write(f.read())
        print("Done.")
    
    # The file does exist, if update is true download it again
    else:
        print("File has already been downloaded.")
        if update:
            print("Checking if updated file exists...")
            file_name_new = os.path.join(os.path.dirname(file_name), ".{}".format(os.path.basename(file_name)))
            file_name_old = os.path.join(os.path.dirname(file_name), "{}_old".format(os.path.basename(file_name)))
            response = urllib.request.urlopen(url)  
        
            f_new = io.BytesIO(response.read())
            with open(file_name_new, 'wb') as outfile:
                outfile.write(f_new.read())
        
            # Check if newly downloaded file is different compared to the previously downloaded one
            # If so, replace the old version with the newer one and save the old version (in case the new one 
            # is broken)
            if not filecmp.cmp(file_name, file_name_new):
                print("File outdated! Replacing with latest version.")
                os.rename(file_name, file_name_old)
                os.rename(file_name_new, file_name)
            else:
                print("Latest version of the file already acquired.")
