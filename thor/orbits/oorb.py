import os
import subprocess
import fortranformat as ff

from ..config import Config
from ..io import readEPHFile

__all__ = ["setupOorb",
           "propagateTestParticle",
           "convertMPCToOorb",
           "propagateOrbits",
           "generateEphemeris"]

def setupOorb(oorbDirectory=Config.oorbDirectory):
    """
    Add oorb installation to PATH.
    
    TODO:
        Check if oorb is already in PATH.
    
    Parameters
    ----------
    oorbDirectory : str, optional
        Path to oorb directory.
        [Default = `~thor.Config.oorbDirectory`]
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
                          configFile=Config.oorbConfigFile,
                          removeFiles=True,
                          verbose=True):
    """
    Propagate a test particle using its ecliptic coordinates and velocity to 
    a given epoch and generate an ephemeris. 
    
    TODO:
        Unittests.
        Use oorb config file.
    
    Parameters
    ----------
    elements : `~np.ndarray` (N, 6)
        Orbital elements in either keplerian or cartesian form. 
        If keplerian:
            Semi-major axis in AU (N, 0).
            Eccentricity (N, 1).
            Inclination in degrees (N, 2).
            Ascending node in degrees (N, 3).
            Argument of perihelion (N, 4).
            Mean anomaly (N, 5).
        If cartesian:
            Ecliptic cartesian coordinates in AU (N, 0:3).
            Ecliptic cartesian velocities in AU per day (N, 3:).
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
        [Default = `~thor.Config.oorbObservatoryCode`]
    configFile : str, optional
        Path to oorb config file which specifies how propagation should
        be done and what orbital element type to output.
        [Default = `~thor.Config.oorbConfigFile`]
    removeFiles : bool, optional
        Clean and remove oorb output files. 
        [Default = True]
    verbose : bool, optional
        Print progress statements?
        
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
    
    # Create file
    file = open(orbInFile, "w")
    for l in header:
        file.write(l)
    file.write(line + "\n")
    file.close()
    
    # Create file
    file = open(orbInFile, "w")
    for l in header:
        file.write(l)
    file.write(line + "\n")
    file.close()
    
    # Propagate orbit
    call = ["oorb",
            "--task=propagation",
            "--orb-in={}".format(orbInFile),
            "--epoch-mjd-utc={}".format(mjdEnd),
            "--orb-out={}".format(orbOutFile),
            "--conf={}".format(configFile)]
    if verbose is True:
        print("Propagating test particle from {} to {}.".format(mjdStart, mjdEnd))
    subprocess.call(call)

    # Generate ephemeris
    if verbose is True:
        print("Generating test particle ephemeris for observatory code {}.".format(observatoryCode))
    ephOut = open(ephOutFile, "w")
    call = ["oorb",
            "--task=ephemeris",
            "--code={}".format(observatoryCode),
            "--orb-in={}".format(orbOutFile),
            "--conf={}".format(configFile)]
    subprocess.call(call, stdout=ephOut)
    ephOut.close()

    # Read ephemeris file
    if verbose is True:
        print("Reading ephemeris file.")
    eph = readEPHFile(ephOutFile)
    
    # Clean files
    if removeFiles is True:
        if verbose is True:
            print("Removing files.")
        for file in [orbInFile, orbOutFile, ephOutFile]:
            os.remove(file)
    
    if verbose is True:
        print("Done.")
        print("")
    return eph

def convertMPCToOorb(mpcorbFile,
                     oorbOutFile,
                     configFile=Config.oorbConfigFile,
                     verbose=True):
    """
    Convert MPCORB file to .orb format for oorb.
    
    TODO:
        Raise errors with oorb.
        Capture stdout, stderr.
    
    Parameters
    ----------
    mpcorbFile : str
        Path to MPCORB.DAT file.
    oorbOutFile : str
        Path to file to which to save newly formatted orbits.
    configFile : str, optional
        Path to oorb config file which specifies how propagation should
        be done and what orbital element type to output.
        [Default = `~thor.Config.oorbConfigFile`]
    verbose : bool, optional
        Print progress statements?
        
    Returns
    -------
    None
    """
    if verbose is True:
        print("Converting {} to {}...".format(mpcorbFile, oorbOutFile))
    
    call = ["oorb",
            "--task=mpcorb",
            "--mpcorb={}".format(mpcorbFile),
            "--orb-out={}".format(oorbOutFile),
            "--conf={}".format(configFile)]
    subprocess.call(call)
    
    if verbose is True:
        print("Done.")
        print("")
    return
    
def propagateOrbits(oorbInFile,
                    oorbOutFile,
                    mjd,
                    configFile=Config.oorbConfigFile,
                    verbose=True):
    """
    Generate ephemeris using oorb.
    
    TODO:
        Raise errors with oorb.
        Capture stdout, stderr.
    
    Parameters
    ----------
    oorbInFile : str
        Path to file with input orbits to propagate.
    oorbOutFile : str
        Path to file to which to save propagated orbits.
    mjd : float
        MJD in UTC to which to propagate input orbits.
    configFile : str, optional
        Path to oorb config file which specifies how propagation should
        be done and what orbital element type to output.
        [Default = `~thor.Config.oorbConfigFile`]
    verbose : bool, optional
        Print progress statements?
        
    Returns
    -------
    None
    """
    if verbose is True:
        print("Propagating {} to {}...".format(oorbInFile, mjd))
        print("Saving to {}.".format(oorbOutFile))
        
    call = ["oorb", 
            "--task=propagation", 
            "--orb-in={}".format(oorbInFile),
            "--epoch-mjd-utc={}".format(mjd),
            "--orb-out={}".format(oorbOutFile),
            "--conf={}".format(configFile)]
    subprocess.call(call)
    
    if verbose is True:
        print("Done.")
        print("")
    return
                
def generateEphemeris(oorbInFile, 
                      ephOutFile, 
                      step=None, 
                      timespan=None,
                      observatoryCode=Config.oorbObservatoryCode, 
                      configFile=Config.oorbConfigFile,
                      verbose=True):
    """
    Generate ephemeris using oorb.
    
    TODO:
        Raise errors with oorb.
        Capture stderr.
    
    Parameters
    ----------
    oorbInFile : str
        Path to file with input orbits from which to generate ephemeris.
    ephOutFile : str
        Path to file to which to save ephemeris output to this file.
    step : float, optional
        If step is defined, generate ephemeris at these steps
        in MJD. Requires timespan to be defined as well.
        [Default = None]
    timespan : float, optional
        If timespan is defined, generate ephemeris at steps in
        MJD for timepan in MJD. 
        [Default = None]
    observatoryCode : str, optional
        Observatory from which to measure ephemerides.
        [Default = `~thor.Config.oorbObservatoryCode`]
    configFile : str, optional
        Path to oorb config file which specifies how propagation should
        be done and what orbital element type to output.
        [Default = `~thor.Config.oorbConfigFile`]
    verbose : bool, optional
        Print progress statements?
        
    Returns
    -------
    None
    """
    if verbose is True:
        print("Generating ephemeris for {} using {}...".format(observatoryCode, oorbInFile))
        if timespan is not None:
            print("Using timespan: {}".format(timespan))
        if step is not None:
            print("Using step: {}".format(step))
        print("Saving to {}.".format(ephOutFile))
        
    call = ["oorb",
            "--task=ephemeris",
            "--code={}".format(observatoryCode),
            "--orb-in={}".format(oorbInFile)]
    
    if timespan is not None:
        call.append("--timespan={}".format(timespan))
            
    if step is not None:
        call.append("--step={}".format(step))
        
    call.append("--conf={}".format(configFile))
                
    subprocess.call(call, stdout=open(ephOutFile, "w"))
    
    if verbose is True:
        print("Done.")
        print("")
    return