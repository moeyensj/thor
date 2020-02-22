import os
import filecmp
import shutil 

from ..codes import getMPCObsCodeFile

def test_getMPCOBSCodeFile():
    # Define some file paths, we expect only obscodes and obscodes_old to be affected 
    # by any function call to getMPCObsCodeFile
    obscodes = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/ObsCodes.html"))
    obscodes_backup = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/ObsCodes_backup.html"))
    obscodes_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/.ObsCodes_test.html"))
    obscodes_old = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/ObsCodes.html_old"))
    
    getMPCObsCodeFile()
    
    # Lets check the file downloaded
    assert os.path.isfile(obscodes)
    
    # Modify the file by removing the last 10 lines
    codes = open(obscodes, "r")
    lines = codes.readlines()
    lines = lines[:-10]
    codes.close()
    
    # Write modified file to a new file
    testfile = open(obscodes_test, "w")
    testfile.writelines(lines)
    testfile.close()
    
    # Rename the modified file to the file name expected by getMPCObsCodeFile
    # See if the function recognizes the change and as a result downloads the correct file again. 
    os.rename(obscodes, obscodes_backup)
    shutil.copy(obscodes_test, obscodes)
    
    getMPCObsCodeFile()
    
    # Check if newly downloaded file is the same as the unmodified one, check that that modified one has been saved as 
    # ObsCodes_old.html
    # (this will only fail in the unlikely chance that between the tests the MPC updated the MPC Obs Codes file)
    assert filecmp.cmp(obscodes, obscodes_backup)
    assert filecmp.cmp(obscodes_test, obscodes_old)
    
    # Clean up files
    os.remove(obscodes_backup)
    os.remove(obscodes_old)
    os.remove(obscodes_test)