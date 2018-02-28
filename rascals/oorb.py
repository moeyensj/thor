import os
import subprocess
import fortranformat as ff

from .config import Config
from .io import readEPHFile

def setupOorb(oorbDirectory=Config.oorbDirectory):
    """
    Add oorb installation to PATH.
    
    TODO:
        Check if oorb is already in PATH.
    
    Parameters
    ----------
    oorbDirectory : str, optional
        Path to oorb directory.
        [Default = `~rascals.Config.oorbDirectory`]
    """
    os.environ["PATH"] += os.pathsep + os.path.join(os.path.abspath(oorbDirectory), "main:")
    os.environ["OORBROOT"] = oorbDirectory
    os.environ["OORB_DATA"] = os.path.join(os.path.abspath(oorbDirectory), "data")
    os.environ["OORB_CONF"] = os.path.join(os.path.abspath(oorbDirectory), "main/oorb.conf")
    os.environ["OORB_GNUPLOTS_SCRIPTS_DIR"] = os.path.join(os.path.abspath(oorbDirectory), "gnuplot")
    return
    
def propagateTestParticle(coords_ec_cart,
                          velocity_ec_cart,
                          mjdStart,
                          mjdEnd,
                          designation="RaSCaL",
                          H=10,
                          G=0.15,
                          observatoryCode=Config.oorbObservatoryCode,
                          removeFiles=True):
    """
    Propagate a test particle using its ecliptic coordinates and velocity to 
    a given epoch and generate an ephemeris. 
    
    TODO:
        Unittests.
        Use oorb config file.
    
    Parameters
    ----------
    coords_ec_cart : `~np.ndarray` (1, 3)
        Ecliptic cartesian coordinates in AU.
    velocity_ec_cart : `~np.ndarray` (1, 3)
        Ecliptic cartesian velocities in AU per day.
    mjdStart : float
        Epoch at which ecliptic coordinates and velocity are measured in MJD.
    mjdEnd : float
        Epoch to which to propagate orbit and generate ephemeris in MJD.
    designation : str, optional
        Name to give test particle. 
        [Default = 'RaSCaL']
    H : float, optional
        Absolute H magnitude.
        [Default = 10]
    G : float, optional
        HG phase function slope. 
        [Default = 0.15]
    observatoryCode : str, optional
        Observatory from which to measure ephemerides.
        [Default = `~rascals.Config.oorbObservatoryCode`]
    removeFiles : bool, optional
        Clean and remove oorb output files. 
        [Default = True]
        
    Returns
    -------
    `~pandas.DataFrame`
        The resulting oorb ephemeris after propagation.
    """
    # Read in the header template
    headerFile = open(os.path.join(os.path.dirname(__file__), "data/header.orb"), "r")
    header = headerFile.readlines()
    header[-1] += "\n"
    
    # Create file path strings
    orbInFile = "testParticle.orb"
    orbOutFile = "testParticle_{}.orb".format(int(mjdEnd))
    ephOutFile = "testParticle_{}.eph".format(int(mjdEnd))
    
    # Create row format
    lineformat = ff.FortranRecordWriter("(A16,6(1X,E21.14),1X,F16.8,1X,F9.5,1X,F9.6)")
    line = lineformat.write([designation, *coords_ec_cart, *velocity_ec_cart, mjdStart, H, G])
    print(line)
    
    # Create file
    file = open(orbInFile, "w")
    for l in header:
        file.write(l)
    file.write(line + "\n")
    file.close()
    
    # Propogate orbit
    call = ["oorb",
            "--task=propagation",
            "--orb-in={}".format(orbInFile),
            "--epoch-mjd-utc={}".format(mjdEnd),
            "--orb-out={}".format(orbOutFile)]
    print("Propagating test particle from {} to {}.".format(mjdStart, mjdEnd))
    subprocess.call(call)

    # Generate ephemeris
    print("Generating test particle ephemeris for observatory code {}.".format(observatoryCode))
    ephOut = open(ephOutFile, "w")
    call = ["oorb",
            "--task=ephemeris",
            "--code={}".format(observatoryCode),
            "--orb-in={}".format(orbOutFile)]
    subprocess.call(call, stdout=ephOut)
    ephOut.close()

    # Read ephemeris file
    print("Reading ephemeris file.")
    eph = readEPHFile(ephOutFile)
    
    # Clean files
    if removeFiles is True:
        print("Removing files.")
        for file in [orbInFile, orbOutFile, ephOutFile]:
            os.remove(file)
    
    print("Done.")
    return eph
      