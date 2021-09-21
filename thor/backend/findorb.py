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

        super().__init__(name="FindOrb", **kwargs)

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

    def _setWorkEnvironment(self, temp_dir):
        # Copy files from user's find_orb directory located in the
        # user's home directory to the .find_orb directory inside
        # the temporary directory
        # Try to ignore files that are created by find_orb when processing
        # observations, also ignore the DE 430 ephemeris file and the asteroid
        # ephemeris file if present
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
                "linux_p1550p2650.430t",
                "asteroid_ephemeris.txt"
            )
        )

        # Create a symlink for the DE 430 file that was not copied
        os.symlink(
            os.path.expanduser("~/.find_orb/linux_p1550p2650.430t"),
            os.path.join(temp_dir, ".find_orb/linux_p1550p2650.430t")
        )

        # If the asteroid ephemeris file is present, also create a symlink for this file
        asteroid_ephem_file = os.path.expanduser("~/.find_orb/asteroid_ephemeris.txt")
        if os.path.exists(asteroid_ephem_file):
            os.symlink(
                asteroid_ephem_file,
                os.path.join(temp_dir, ".find_orb/asteroid_ephemeris.txt")
            )

        # Create a copy of the environment and set the HOME directory to the
        # temporary directory
        env = os.environ.copy()
        env["HOME"] = temp_dir
        return env

    def _propagateOrbits(self, orbits, t1):
        """


        """
        propagated_dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:

            # Set this environment's home directory to the temporary directory
            # to prevent bugs with multiple processes writing to the ~/.find_orb/
            # directory
            env = self._setWorkEnvironment(temp_dir)

            # Write the desired times out to a file that find_orb understands
            times_in_file = os.path.join(temp_dir, "times_prop.in")
            self._writeTimes(
                times_in_file,
                t1.tt,
                "tt"
            )

            for i in range(orbits.num_orbits):

                # If you give fo a string for a numbered object it will typically append brackets
                # automatically which makes retrieving the object's orbit a little more tedious so by making sure
                # the object ID is not numeric we can work around that.
                orbit_id_i = f"o{i:08d}"
                out_dir = os.path.join(temp_dir, "vectors{}".format(orbit_id_i))
                vectors_txt = os.path.join(out_dir, "vectors.txt")

                # Format the state vector into the type understood by find_orb
                fo_orbit = "-v{},{}".format(
                    orbits.epochs[i].tt.jd,
                    ",".join(orbits.cartesian[i].astype("str"))
                )

                # Run fo
                call = [
                    "fo",
                    "-o",
                    "o{}".format(orbit_id_i),
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
                    cwd=temp_dir,
                    check=False,
                    capture_output=True
                )

                if (ret.returncode != 0):
                    warning = (
                        "fo returned a non-zero error code./n",
                        "Command: /n",
                        " ".join(call) + "/n",
                        ret.stdout.decode('utf-8'),
                        ret.stderr.decode('utf-8')
                    )
                    warnings.warn(warning)

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

                propagated_dfs.append(df)

            propagated = pd.concat(propagated_dfs, ignore_index=True)
            propagated["mjd_tdb"] = Time(
                propagated["jd_tt"].values,
                scale="tt",
                format="jd"
            ).tdb.mjd
            propagated = propagated[["orbit_id", "mjd_tdb", "x", "y", "z", "vx", "vy", "vz"]]

            if orbits.ids is not None:
                propagated["orbit_id"] = orbits.ids[propagated["orbit_id"].values]

        return propagated

    def _generateEphemeris(self, orbits, observers):
        ephemeris_dfs = []
        with tempfile.TemporaryDirectory() as temp_dir:

            # Set this environment's home directory to the temporary directory
            # to prevent bugs with multiple processes writing to the ~/.find_orb/
            # directory
            env = self._setWorkEnvironment(temp_dir)

            for observatory_code, observation_times in observers.items():

                # Write the desired times out to a file that find_orb understands
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

                    # If you give fo a string for a numbered object it will typically append brackets
                    # automatically which makes retrieving the object's orbit a little more tedious so by making sure
                    # the object ID is not numeric we can work around that.
                    orbit_id_i = f"o{i:08d}"
                    out_dir = os.path.join(temp_dir, "ephemeris{}_{}".format(orbit_id_i, observatory_code))
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
                        "o{}".format(orbit_id_i),
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
                        cwd=temp_dir,
                        check=False,
                        capture_output=True
                    )

                    if (os.path.exists(ephemeris_txt)):
                        ephemeris = pd.read_csv(
                            ephemeris_txt,
                            header=0,
                            delim_whitespace=True,
                            names=columns,
                            float_precision="round_trip"

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
        residual_dfs = []

        # Find_Orb accepts ADES files as inputs for orbit determination so
        # lets convert THOR-like observations into essentially dummy
        # ADES files that find_orb can process
        _observations = observations.copy()
        _observations.rename(
            columns={
                "mjd_utc" : "mjd",
                "RA_deg" : "ra",
                "Dec_deg" : "dec",
                "RA_sigma_deg" : "rmsRA",
                "Dec_sigma_deg" : "rmsDec",
                "mag_sigma" : "rmsMag",
                "mjd_sigma_seconds" : "rmsTime",
                "filter" : "band",
                "observatory_code" : "stn",
            },
            inplace=True
        )
        _observations["rmsRA"] = _observations["rmsRA"].values * 3600
        _observations["rmsDec"] = _observations["rmsDec"].values * 3600
        _observations.sort_values(
            by=["mjd", "stn"],
            inplace=True
        )

        id_present = False
        id_col = None
        if "permID" in _observations.columns.values:
            id_present = True
            id_col = "permID"
        if "provID" in _observations.columns.values:
            id_present = True
            id_col = "provID"
        if "trkSub" in _observations.columns.values:
            id_present = True
            id_col = "trkSub"
        if not id_present:
            _observations["trkSub"] = _observations["orbit_id"]
            id_col = "trkSub"

        if "mag" not in _observations.columns:
            _observations.loc[:, "mag"] = 20.0
        if "band" not in _observations.columns:
            _observations.loc[:, "band"] = "V"
        if "mode" not in _observations.columns:
            _observations.loc[:, "mode"] = "CCD"
        if "astCat" not in _observations.columns:
            _observations.loc[:, "astCat"] = "None"


        for i, orbit_id in enumerate(_observations[id_col].unique()):

            with tempfile.TemporaryDirectory() as temp_dir:

                # Set this environment's home directory to the temporary directory
                # to prevent bugs with multiple processes writing to the ~/.find_orb/
                # directory
                env = self._setWorkEnvironment(temp_dir)

                # If you give fo a string for a numbered object it will typically append brackets
                # automatically which makes retrieving the object's orbit a little more tedious so by making sure
                # the object ID is not numeric we can work around that.
                orbit_id_i = f"o{i:08d}"

                observations_file = os.path.join(temp_dir, "_observations_{}.psv".format(orbit_id_i))
                out_dir = os.path.join(temp_dir, "od_{}".format(orbit_id_i))

                mask = _observations[id_col].isin([orbit_id])
                object_observations = _observations[mask].copy()
                object_observations.loc[:, id_col] = orbit_id_i
                object_observations.reset_index(inplace=True, drop=True)

                writeToADES(
                    object_observations,
                    observations_file,
                    mjd_scale="utc"
                )

                last_observation = Time(
                    object_observations["mjd"].max(),
                    scale="utc",
                    format="mjd"
                )
                call = [
                    "fo",
                    observations_file,
                    "-O",
                    out_dir,
                    "-tEjd{}".format(last_observation.tt.jd),
                    "-j",
                    "-D",
                    self.config_file,
                ]

                ret = subprocess.run(
                    call,
                    shell=False,
                    env=env,
                    cwd=temp_dir,
                    check=False,
                    capture_output=True
                )

                covar_json = os.path.join(out_dir, "covar.json")
                if (os.path.exists(covar_json)) and ret.returncode == 0:
                    with open(covar_json) as f:
                        covar_data = json.load(f)
                        epoch = covar_data["epoch"]
                        state = np.array(covar_data["state_vect"])
                        covariance_matrix = np.array(covar_data["covar"])
                else:
                    epoch = np.NaN
                    state = np.zeros(6) * np.NaN
                    covariance_matrix = np.zeros((6,6)) * np.NaN

                total_json = os.path.join(out_dir, "total.json")
                if (os.path.exists(total_json)) and ret.returncode == 0:
                    with open(total_json) as f:
                        data = json.load(f)
                        residuals = pd.DataFrame(
                            data["objects"][orbit_id_i]["observations"]["residuals"]
                        )
                        residuals.sort_values(
                            by=["JD", "obscode"],
                            inplace=True
                        )
                        residuals = object_observations[["obs_id"]].join(residuals)

                    residual_dfs.append(residuals)

                ids.append(orbit_id)
                epochs.append(epoch)
                orbits.append(state)
                covariances.append(covariance_matrix)

        orbits = np.vstack(orbits)

        od_orbits = pd.DataFrame({
            "orbit_id" : ids,
            "jd_tt" : epochs,
            "x" : orbits[:, 0],
            "y" : orbits[:, 1],
            "z" : orbits[:, 2],
            "vx" : orbits[:, 3],
            "vy" : orbits[:, 4],
            "vz" : orbits[:, 5],
            "covariance" : covariances
        })

        od_orbits["mjd_tdb"] = np.NaN
        od_orbits.loc[~od_orbits["jd_tt"].isna(), "mjd_tdb"] = Time(
            od_orbits[~od_orbits["jd_tt"].isna()]["jd_tt"].values,
            scale="tt",
            format="jd"
        ).tdb.mjd

        od_orbits = od_orbits[[
            "orbit_id",
            "mjd_tdb",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "covariance"
        ]]
        od_orbit_members = pd.concat(residual_dfs, ignore_index=True)

        return od_orbits, od_orbit_members