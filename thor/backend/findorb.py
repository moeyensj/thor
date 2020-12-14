import os
import time
import json
import shutil
import tempfile
import warnings
import subprocess 
import numpy as np
import pandas as pd
from astropy.time import Time

from ..utils import writeToADES
from .backend import Backend

FINDORB_CONFIG = {
    "config_file" : os.path.join(os.path.dirname(__file__), "data", "environ.dat"),
    "remove_files" : True,
}

class FINDORB(Backend):
    
    def __init__(self, **kwargs):
        
        # Make sure only the correct kwargs
        # are passed to the constructor
        allowed_kwargs = FINDORB_CONFIG.keys()
        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError()
        
        # If an allowed kwarg is missing, add the 
        # default 
        for k in allowed_kwargs:
            if k not in kwargs:
                kwargs[k] = FINDORB_CONFIG[k]
        
        super(FINDORB, self).__init__(**kwargs)

        return
    
    def _writeTimes(self, file_name, times, time_scale):
        """
        Write a time file that find_orb can use to propagate orbits
        and generate ephemerides. 

        Parameters
        ----------
        file_name : str
            Path of desired file.
        times :
            Times to write into the file
        time_scale : str
            Time scale to which to ouput to file. 
        """
        with open(file_name, "w") as f:
            f.write("OPTION {}\n".format(time_scale.upper()))
            f.write("OPTION ASCENDING\n")
            np.savetxt(
                f, 
                times.jd, 
                newline="\n", 
                fmt="%.16f"
            )
            f.close()
        return 

    def _setWorkEnv(self, temp_dir):

        shutil.copytree(
            os.path.expanduser("~/.find_orb"), 
            os.path.join(temp_dir, ".find_orb"),
            ignore=shutil.ignore_patterns(
                "elements.txt",
                "guide.txt",
                "mpc_fmt.txt",
                "observe.txt",
                "residual.txt",
                "total.json",
                "vectors.txt",
                "covar.txt",
                "debug.txt", 
                "elements.json", 
                "elem_short.json", 
                "combined.json",
                "linux_p1550p2650.430t"
            )
        )
        os.symlink(
            os.path.expanduser("~/.find_orb/linux_p1550p2650.430t"), 
            os.path.join(temp_dir, ".find_orb/linux_p1550p2650.430t")
        )

        env = os.environ.copy()
        env["HOME"] = temp_dir
        return env

    def _propagateOrbits(self, orbits, t1):
        """
        Propagate orbits, represented as cartesian state vectors, define at t0
        to times t1. 




        """
        propagated_dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Set this environment's home directory to the temporary directory
            # to prevent bugs with multiple processes writing to the ~/.find_orb/
            # directory
            env = self._setWorkEnv(temp_dir)
            
            times_in_file = os.path.join(temp_dir, "times_prop.in")
            self._writeTimes(
                times_in_file,
                t1.tt,
                "tt"
            )
            
            for i in range(orbits.num_orbits):
                    
                orbit_id = orbits.ids[i]
                out_dir = os.path.join(temp_dir, "vectors{}".format(orbit_id))
                vectors_txt = os.path.join(out_dir, "vectors.txt")

                # Format the state vector into the type understood by find_orb
                fo_orbit = "-v{},{}".format(
                    orbits.epochs[i].tt.jd,
                    ",".join(orbits.cartesian[i].astype("str"))
                )

                call = [
                    "fo", 
                    "-o", 
                    "o{}".format(orbit_id), 
                    fo_orbit,
                    "-E",
                    "0",
                    "-C",
                    "Sun",
                    "-O",
                    out_dir,
                    "-e",
                    vectors_txt,
                    "-D",
                    self.config_file, 
                    "EPHEM_STEP_SIZE=t{}".format(times_in_file)
                ]
                ret = subprocess.run(
                    call, 
                    shell=False, 
                    env=env, 
                    check=False, 
                    capture_output=True
                )
                
                if (os.path.exists(vectors_txt)):
                    df = pd.read_csv(
                        vectors_txt,
                        header=0,
                        delim_whitespace=True,
                        names=["jd_tt", "x", "y", "z", "vx", "vy", "vz"]
                    )
                    df["orbit_id"] = [i for _ in range(len(df))]
                else:
                    df = pd.DataFrame(
                        columns=["orbit_id", "jd_tt", "x", "y", "z", "vx", "vy", "vz"]
                    )

                #result["name"] = trkSub
                #result["findorb"] = {
                #    'args': ret.args,
                #    'returncode': ret.returncode,
                #    'stdout': ret.stdout.decode('utf-8'),
                #    'stderr': ret.stderr.decode('utf-8')
                #}
                #print(ret.stdout.decode('utf-8'))
                #print(ret.stderr.decode('utf-8'))
                propagated_dfs.append(df)

            propagated = pd.concat(propagated_dfs)
            propagated.reset_index(
                inplace=True,
                drop=True
            )
            propagated["epoch_mjd_tdb"] = Time(
                propagated["jd_tt"].values,
                scale="tt",
                format="jd"
            ).tdb.mjd
            propagated = propagated[["orbit_id", "epoch_mjd_tdb", "x", "y", "z", "vx", "vy", "vz"]]

            if orbits.ids is not None:
                propagated["orbit_id"] = orbits.ids[propagated["orbit_id"].values]

        return propagated

    def _generateEphemeris(self, orbits, observers):

        ephemeris_dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Set this environment's home directory to the temporary directory
            # to prevent bugs with multiple processes writing to the ~/.find_orb/
            # directory
            env = self._setWorkEnv(temp_dir)
            
            for observatory_code, observation_times in observers.items():
            
                times_in_file = os.path.join(temp_dir, "times_eph.in")
                self._writeTimes(
                    times_in_file,
                    observation_times.tt,
                    "tt"
                )

                # Certain observatories are dealt with differently, so appropriately
                # define the columns
                if (observatory_code == "500"):
                    columns = [
                        "jd_utc", "RA_deg", "Dec_deg", "delta_au", "r_au",
                        "elong", "ph_ang", "mag",  
                        "'/hr", "PA", "rvel", 
                        "lon", "lat", "altitude_km"
                    ] 
                else:
                    columns = [
                        "jd_utc", "RA_deg", "Dec_deg", "delta_au", "r_au",
                        "elong", "ph_ang", "mag",  
                        "RA_'/hr", "dec_'/hr",  
                        "alt",  "az",  "rvel",   
                        "lon", "lat", "altitude_km"
                    ] 

                # For each orbit calculate their ephemerides
                for i in range(orbits.num_orbits):

                    # Set directory variables
                    orbit_id = orbits.ids[i]
                    out_dir = os.path.join(temp_dir, "ephemeris{}_{}".format(orbit_id, observatory_code))
                    ephemeris_txt = os.path.join(out_dir, "ephemeris.txt")
                    
                    # Format the state vector into the type understood by find_orb
                    fo_orbit = "-v{},{}".format(
                        orbits.epochs[i].tt.jd,
                        ",".join(orbits.cartesian[i].astype("str"))
                    )
                    if orbits.H is not None:
                        fo_orbit += ",H={}".format(orbits.H[i])

                    # Call fo and generate ephemerides
                    call = [
                        "fo", 
                        "-o", 
                        "o{}".format(orbit_id), 
                        fo_orbit,
                        "-C",
                        "{}".format(observatory_code),
                        "-O",
                        out_dir,
                        "-e",
                        ephemeris_txt,
                        "-D",
                        self.config_file,
                        "EPHEM_STEP_SIZE=t{}".format(times_in_file),   
                        "JSON_EPHEM_NAME={}".format(os.path.join(temp_dir, "eph%p_%c.json")),
                        "JSON_ELEMENTS_NAME={}".format(os.path.join(temp_dir, "ele%p.json")),
                        "JSON_SHORT_ELEMENTS={}".format(os.path.join(temp_dir, "short%p.json")),
                        "JSON_COMBINED_NAME={}".format(os.path.join(temp_dir, "com%p_%c.json"))
                    ]
                    ret = subprocess.run(
                        call, 
                        shell=False, 
                        env=env, 
                        check=False, 
                        capture_output=False
                    )
                    
                    if (os.path.exists(ephemeris_txt)):
                        ephemeris = pd.read_csv(
                            ephemeris_txt,
                            header=0,
                            delim_whitespace=True,
                            names=columns

                        )
                        ephemeris["orbit_id"] = [i for _ in range(len(ephemeris))]
                        ephemeris["observatory_code"] = [observatory_code for _ in range(len(ephemeris))]
                    else:
                        ephemeris = pd.DataFrame(
                            columns=[["orbit_id", "observatory_code"] + columns]
                        )

                    ephemeris_dfs.append(ephemeris)

        # Combine ephemeris data frames and sort by orbit ID,
        # observatory code and observation time, then reset the
        # index
        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.sort_values(
            by=["orbit_id", "observatory_code", "jd_utc"],
            inplace=True
        )
        ephemeris.reset_index(
            inplace=True,
            drop=True
        )

        # Extract observation times and convert them to MJDs
        times = Time(
            ephemeris["jd_utc"].values,
            format="jd",
            scale="utc"
        )
        ephemeris["mjd_utc"] = times.utc.mjd
        ephemeris.drop(
            columns=["jd_utc"],
            inplace=True
        )

        # Sort the columns into a more user friendly order
        cols = ['orbit_id', 'observatory_code', 'mjd_utc'] + [col for col in ephemeris.columns if col not in ['orbit_id', 'observatory_code', 'mjd_utc']]
        ephemeris = ephemeris[cols]

        # If orbits have their IDs defined replace the orbit IDs
        if orbits.ids is not None:
            ephemeris["orbit_id"] = orbits.ids[ephemeris["orbit_id"].values]

        return ephemeris

    def _orbitDetermination(self, observations):

        ids = []
        epochs = []
        orbits = []
        covariances = []

        _observations = observations.copy()
        _observations.rename(
            columns={
                "RA_deg" : "ra",
                "Dec_deg" : "dec",
                "RA_sigma_deg" : "rmsRA",
                "Dec_sigma_deg" : "rmsDec",
                "mag_sigma" : "rmsMag",
                "filter" : "band",
                "observatory_code" : "stn"
            }, 
            inplace=True
        )
        _observations["rmsRA"] = _observations["rmsRA"] * np.cos(np.radians(_observations["dec"].values)) / 3600
        _observations["rmsDec"] = _observations["rmsDec"] / 3600

        id_present = False
        if "permID" in _observations.columns.values:
            id_present = True
        if "provID" in _observations.columns.values:
            id_present = True
        if "trkSub" in _observations.columns.values:
            id_present = True
        if not id_present:
            _observations["trkSub"] = _observations["obj_id"]
        
        if "mag" not in _observations.columns:
            _observations.loc[:, "mag"] = 20.0
        if "band" not in _observations.columns:
            _observations.loc[:, "band"] = "V"
        if "mode" not in _observations.columns:
            _observations.loc[:, "mode"] = "CCD"
        if "astCat" not in _observations.columns:
            _observations.loc[:, "astCat"] = "None"

        with tempfile.TemporaryDirectory() as temp_dir:

            # Set this environment's home directory to the temporary directory
            # to prevent bugs with multiple processes writing to the ~/.find_orb/
            # directory
            env = self._setWorkEnv(temp_dir)

            for obj_id in _observations["obj_id"].unique():

                observations_file = os.path.join(temp_dir, "_observations_{}.psv".format(obj_id))
                out_dir = os.path.join(temp_dir, "od_{}".format(obj_id))

                mask = _observations["obj_id"].isin([obj_id])
                writeToADES(
                    _observations[mask], 
                    observations_file, 
                    mjd_scale="utc"
                )

                call = [
                    "fo", 
                    observations_file, 
                    "-O", 
                    out_dir,
                    "-D",
                    self.config_file,
                ]
                ret = subprocess.run(
                        call, 
                        shell=False, 
                        env=env, 
                        check=False, 
                        capture_output=True
                )
                
                covar_json = os.path.join(out_dir, "covar.json")
                if (os.path.exists(covar_json)):
                    with open(os.path.join(out_dir, "covar.json")) as f:
                        covar_data = json.load(f)
                        epoch = covar_data["epoch"]
                        state = np.array(covar_data["state_vect"])
                        covariance_matrix = np.array(covar_data["covar"])
                else:
                    epoch = 0.0
                    state = np.zeros(6, dtype=float)
                    covariance_matrix = np.zeros((6, 6), dtype=float)

                ids.append(obj_id)
                epochs.append(epoch)
                orbits.append(state)
                covariances.append(covariance_matrix)

        orbits = np.vstack(orbits)

        od = pd.DataFrame({
            "obj_id" : ids,
            "jd_tt" : epochs,
            "x" : orbits[:, 0],
            "y" : orbits[:, 1],
            "z" : orbits[:, 2],
            "vx" : orbits[:, 3],
            "vy" : orbits[:, 4],
            "vz" : orbits[:, 5],
            "covariance" : covariances
        })
        
        od["mjd_tdb"] = Time(
            od["jd_tt"].values,
            format="jd",
            scale="tt"
        ).tdb.mjd
        
        od = od[[
            "obj_id", 
            "mjd_tdb", 
            "x", 
            "y", 
            "z", 
            "vx", 
            "vy", 
            "vz", 
            "covariance"
        ]]
        
        return od

