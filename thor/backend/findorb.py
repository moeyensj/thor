import os
import shutil
import warnings
import subprocess 
import numpy as np
import pandas as pd
from astropy.time import Time

from .backend import Backend

FINDORB_CONFIG = {
    "remove_files" : "N",
}
FINDORB_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "data", "environ.dat")

class FINDORB(Backend):
    
    def __init__(self, **kwargs):
        
        allowed_kwargs = [
            "remove_files", 
        ]
        
        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError()
        
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
        return 

    def propagateOrbits(self, orbits, t1):
        """
        Propagate orbits, represented as cartesian state vectors, define at t0
        to times t1. 




        """

        self._writeTimes(
            "times_prop.in",
            t1.tt,
            "tt"
        )

        dfs = []
        for i in range(orbits.num_orbits):

            out_dir = "vectors{:08d}".format(i)
            vectors_txt = "{}/vectors.txt".format(out_dir)
            call = [
                "fo", 
                "-o", 
                "o{:08d}".format(i), 
                "-v",  
                "{},{}".format(
                    orbits.epochs[i].tt.jd,
                    ",".join(orbits.cartesian[i].astype("str"))
                ),
                "-E",
                "0",
                "-C",
                "Sun",
                "-O",
                out_dir,
                "-e",
                vectors_txt,
                "-D",
                FINDORB_CONFIG_FILE, 
                "EPHEM_STEP_SIZE=ttimes_prop.in"

            ]
            subprocess.call(call)


            # Read results of propagation                
            df = pd.read_csv(
                vectors_txt,
                header=0,
                delim_whitespace=True,
                names=["jd_tt", "x", "y", "z", "vx", "vy", "vz"]
            )
            df["orbit_id"] = [i for _ in range(len(df))]

            if self.remove_files:
                shutil.rmtree(out_dir)

            dfs.append(df)

        propagated = pd.concat(dfs)
        propagated.reset_index(
            inplace=True,
            drop=True
        )
        propagated = propagated[["orbit_id", "jd_tt", "x", "y", "z", "vx", "vy", "vz"]]

        if self.remove_files:
            os.remove("times_prop.in")

        if orbits.ids is not None:
            propagated["orbit_id"] = orbits.ids[propagated["orbit_id"].values]

        return propagated

    def generateEphemeris(self, orbits, observers):

        ephemeris_dfs = []
        for observatory_code, observation_times in observers.items():

            self._writeTimes(
                "times_eph.in",
                observation_times.utc,
                "utc"
            )

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


            for i in range(orbits.num_orbits):

                out_dir = "ephemeris{:08d}_{}".format(i, observatory_code)
                ephemeris_txt = "{}/ephemeris.txt".format(out_dir)
                print(FINDORB_CONFIG_FILE)
                call = [
                    "fo", 
                    "-o", 
                    "o{:08d}".format(i), 
                    "-v",  
                    "{},{},H=20.0".format(
                        orbits.epochs[i].tt.jd,
                        ",".join(orbits.cartesian[i].astype("str")),
                    ),
                    "-C",
                    "{}".format(observatory_code),
                    "-O",
                    out_dir,
                    "-e",
                    ephemeris_txt,
                    "-D",
                    FINDORB_CONFIG_FILE,
                    "EPHEM_STEP_SIZE=ttimes_eph.in",   
                ]
                subprocess.call(call)

                ephemeris = pd.read_csv(
                    ephemeris_txt,
                    header=0,
                    delim_whitespace=True,
                    names=columns

                )
                ephemeris["orbit_id"] = [i for _ in range(len(ephemeris))]
                ephemeris["observatory_code"] = [observatory_code for _ in range(len(ephemeris))]
                ephemeris_dfs.append(ephemeris)

                if self.remove_files:
                    shutil.rmtree(out_dir)
                    os.remove("eph_json.txt")

            if self.remove_files:
                os.remove("times_eph.in")


        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.sort_values(
            by=["orbit_id", "observatory_code", "jd_utc"],
            inplace=True
        )
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
        ephemeris.reset_index(
            inplace=True,
            drop=True
        )

        cols = ['orbit_id', 'observatory_code', 'mjd_utc'] + [col for col in ephemeris.columns if col not in ['orbit_id', 'observatory_code', 'mjd_utc']]
        ephemeris = ephemeris[cols]

        if orbits.ids is not None:
            ephemeris["orbit_id"] = orbits.ids[ephemeris["orbit_id"].values]

        return ephemeris


def orbitDetermination(observations):
    
    ids = []
    epochs = []
    orbits = []
    covariances = []
    
    for obj_id in observations["obj_id"].unique():
        
        mask = observations["obj_id"].isin([obj_id])
        writeToADES(
            observations[mask], 
            "observations.psv", 
            mjd_scale="utc"
        )
        
        out_dir = "od_{}".format(obj_id)
        call = [
            "fo", 
            "observations.psv", 
            "-O", 
            out_dir,
            "-D",
            "environ.dat",
        ]
        subprocess.call(call)

        with open(os.path.join(out_dir, "covar.json")) as f:
            covar_data = json.load(f)
            covariance_matrix = np.array(covar_data["covar"])
            state = np.array(covar_data["state_vect"])
            epoch = covar_data["epoch"]

        ids.append(obj_id)
        epochs.append(epoch)
        orbits.append(state)
        covariances.append(covariance_matrix)
        
        if remove_files:
            shutil.rmtree(out_dir)
            os.remove("observations.psv")
        
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
    return od

