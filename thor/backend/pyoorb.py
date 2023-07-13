try:
    import pyoorb as oo
except ImportError:
    raise ImportError("PYOORB is not installed.")

import enum
import os
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from astropy.time import Time

from .backend import Backend


class OpenOrbTimescale(enum.Enum):
    UTC = 1
    UT1 = 2
    TT = 3
    TAI = 4


class OpenOrbOrbitType(enum.Enum):
    CARTESIAN = 1
    COMETARY = 2
    KEPLERIAN = 3


class PYOORB(Backend):
    def __init__(
        self, *, dynamical_model: str = "N", ephemeris_file: str = "de430.dat"
    ):

        self.dynamical_model = dynamical_model
        self.ephemeris_file = ephemeris_file

        env_var = "ADAM_CORE_PYOORB_INITIALIZED"
        if env_var in os.environ.keys() and os.environ[env_var] == "True":
            pass
        else:
            if os.environ.get("OORB_DATA") is None:
                if os.environ.get("CONDA_PREFIX") is None:
                    raise RuntimeError(
                        "Cannot find OORB_DATA directory. Please set the OORB_DATA environment variable."
                    )
                else:
                    os.environ["OORB_DATA"] = os.path.join(
                        os.environ["CONDA_PREFIX"], "share/openorb"
                    )

            oorb_data = os.environ["OORB_DATA"]

            # Prepare pyoorb
            ephfile = os.path.join(oorb_data, self.ephemeris_file)
            err = oo.pyoorb.oorb_init(ephfile)
            if err == 0:
                os.environ[env_var] = "True"
            else:
                warnings.warn(f"PYOORB returned error code: {err}")

        return

    @staticmethod
    def _configure_orbits(
        orbits: np.ndarray,
        t0: np.ndarray,
        orbit_type: OpenOrbOrbitType,
        time_scale: OpenOrbTimescale,
        magnitude: Optional[Union[float, np.ndarray]] = None,
        slope: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to convert. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        orbit_type : OpenOrbOrbitType
            Orbital element representation of the provided orbits.
            If cartesian:
                x : heliocentric ecliptic J2000 x position in AU
                y : heliocentric ecliptic J2000 y position in AU
                z : heliocentric ecliptic J2000 z position in AU
                vx : heliocentric ecliptic J2000 x velocity in AU per day
                vy : heliocentric ecliptic J2000 y velocity in AU per day
                vz : heliocentric ecliptic J2000 z velocity in AU per day
            If keplerian:
                a : semi-major axis in AU
                e : eccentricity in degrees
                i : inclination in degrees
                Omega : longitude of the ascending node in degrees
                omega : argument of periapsis in degrees
                M0 : mean anomaly in degrees
            If cometary:
                p : perihelion distance in AU
                e : eccentricity in degrees
                i : inclination in degrees
                Omega : longitude of the ascending node in degrees
                omega : argument of periapsis in degrees
                T0 : time of perihelion passage in MJD
        time_scale : OpenOrbTimescale
            Time scale of the MJD epochs.
        magnitude : float or `~numpy.ndarray` (N), optional
            Absolute H-magnitude or M1 magnitude.
        slope : float or `~numpy.ndarray` (N), optional
            Photometric slope parameter G or K1.

        Returns
        -------
        orbits_pyoorb : `~numpy.ndarray` (N, 12)
            Orbits formatted in the format expected by PYOORB.
                orbit_id : index of input orbits
                elements x6: orbital elements of propagated orbits
                orbit_type : orbit type
                epoch_mjd : epoch of the propagate orbit
                time_scale : time scale of output epochs
                H/M1 : absolute magnitude
                G/K1 : photometric slope parameter
        """
        orbits_ = orbits.copy()
        num_orbits = orbits_.shape[0]

        orbit_type_ = np.array([orbit_type.value for i in range(num_orbits)])

        time_scale_ = np.array([time_scale.value for i in range(num_orbits)])

        if isinstance(slope, (float, int)):
            slope_ = np.array([slope for i in range(num_orbits)])
        elif isinstance(slope, list):
            slope_ = np.array(slope)
        elif isinstance(slope, np.ndarray):
            slope_ = slope
        else:
            slope_ = np.array([0.15 for i in range(num_orbits)])

        if isinstance(magnitude, (float, int)):
            magnitude_ = np.array([magnitude for i in range(num_orbits)])
        elif isinstance(magnitude, list):
            magnitude_ = np.array(magnitude)
        elif isinstance(magnitude, np.ndarray):
            magnitude_ = magnitude
        else:
            magnitude_ = np.array([20.0 for i in range(num_orbits)])

        ids = np.array([i for i in range(num_orbits)])

        orbits_pyoorb = np.zeros((num_orbits, 12), dtype=np.double, order="F")
        orbits_pyoorb[:, 0] = ids
        orbits_pyoorb[:, 1:7] = orbits_
        orbits_pyoorb[:, 7] = orbit_type_
        orbits_pyoorb[:, 8] = t0
        orbits_pyoorb[:, 9] = time_scale_
        orbits_pyoorb[:, 10] = magnitude_
        orbits_pyoorb[:, 11] = slope_

        return orbits_pyoorb

    @staticmethod
    def _configure_epochs(
        epochs: np.ndarray, time_scale: OpenOrbTimescale
    ) -> np.ndarray:
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        epochs : `~numpy.ndarray` (N)
            Epoch in MJD to convert.
        time_scale : OpenOrbTimescale
            Time scale of the MJD epochs.

        Returns
        -------
        epochs_pyoorb : `~numpy.ndarray` (N, 2)
            Epochs converted into the PYOORB format.
        """
        num_times = len(epochs)
        time_scale_list = [time_scale.value for i in range(num_times)]
        epochs_pyoorb = np.array(
            list(np.vstack([epochs, time_scale_list]).T), dtype=np.double, order="F"
        )
        return epochs_pyoorb

    def _propagateOrbits(self, orbits, t1):
        """
        Propagate orbits using PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to propagate. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        t1 : `~numpy.ndarray` (N)
            Epoch in MJD to which to propagate the orbits.
        orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
            Heliocentric ecliptic J2000 orbital element representation of the provided orbits
            If 'cartesian':
                x : x-position [AU]
                y : y-position [AU]
                z : z-position [AU]
                vx : x-velocity [AU per day]
                vy : y-velocity [AU per day]
                vz : z-velocity [AU per day]
            If 'keplerian':
                a : semi-major axis [AU]
                e : eccentricity [degrees]
                i : inclination [degrees]
                Omega : longitude of the ascending node [degrees]
                omega : argument of periapsis [degrees]
                M0 : mean anomaly [degrees]
            If 'cometary':
                p : perihelion distance [AU]
                e : eccentricity [degrees]
                i : inclination [degrees]
                Omega : longitude of the ascending node [degrees]
                omega : argument of periapsis [degrees]
                T0 : time of perihelion passage [degrees]
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
            Time scale of the MJD epochs.
        magnitude : float or `~numpy.ndarray` (N), optional
            Absolute H-magnitude or M1 magnitude.
        slope : float or `~numpy.ndarray` (N), optional
            Photometric slope parameter G or K1.
        dynamical_model : {'N', '2'}, optional
            Propagate using N or 2-body dynamics.
        ephemeris_file : str, optional
            Which JPL ephemeris file to use with PYOORB.

        Returns
        -------
        propagated : `~pandas.DataFrame`
            Orbits at new epochs.
        """
        # Convert orbits into PYOORB format
        orbits_pyoorb = self._configure_orbits(
            orbits.cartesian,
            orbits.epochs.tt.mjd,
            OpenOrbOrbitType.CARTESIAN,
            OpenOrbTimescale.TT,
            magnitude=None,
            slope=None,
        )

        # Convert epochs into PYOORB format
        epochs_pyoorb = self._configure_epochs(t1.tt.mjd, OpenOrbTimescale.TT)

        # Propagate orbits to each epoch and append to list
        # of new states
        states = []
        orbits_pyoorb_i = orbits_pyoorb.copy()
        for epoch in epochs_pyoorb:
            orbits_pyoorb_i, err = oo.pyoorb.oorb_propagation(
                in_orbits=orbits_pyoorb_i,
                in_epoch=epoch,
                in_dynmodel=self.dynamical_model,
            )
            states.append(orbits_pyoorb_i)

        # Convert list of new states into a pandas data frame
        # These states at the moment will always be return as cartesian
        # state vectors
        elements = ["x", "y", "z", "vx", "vy", "vz"]
        # Other PYOORB state vector representations:
        # "keplerian":
        #    elements = ["a", "e", "i", "Omega", "omega", "M0"]
        # "cometary":
        #    elements = ["q", "e", "i", "Omega", "omega", "T0"]

        # Create pandas data frame
        columns = [
            "orbit_id",
            *elements,
            "orbit_type",
            "epoch_mjd",
            "time_scale",
            "H/M1",
            "G/K1",
        ]
        propagated = pd.DataFrame(np.concatenate(states), columns=columns)
        propagated["orbit_id"] = propagated["orbit_id"].astype(int)

        # Convert output epochs to TDB
        epochs = Time(propagated["epoch_mjd"].values, format="mjd", scale="tt")
        propagated["mjd_tdb"] = epochs.tdb.value

        # Drop PYOORB specific columns (may want to consider this option later on.)
        propagated.drop(
            columns=["epoch_mjd", "orbit_type", "time_scale", "H/M1", "G/K1"],
            inplace=True,
        )

        # Re-order columns and sort
        propagated = propagated[["orbit_id", "mjd_tdb"] + elements]
        propagated.sort_values(
            by=["orbit_id", "mjd_tdb"], inplace=True, ignore_index=True
        )

        if orbits.ids is not None:
            propagated["orbit_id"] = orbits.ids[propagated["orbit_id"].values]

        return propagated

    def _generateEphemeris(self, orbits, observers):
        """
        Generate ephemeris using PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to propagate. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        t1 : `~numpy.ndarray` (N)
            Epoch in MJD to which to propagate the orbits.
        orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
            Heliocentric ecliptic J2000 orbital element representation of the provided orbits
            If 'cartesian':
                x : x-position [AU]
                y : y-position [AU]
                z : z-position [AU]
                vx : x-velocity [AU per day]
                vy : y-velocity [AU per day]
                vz : z-velocity [AU per day]
            If 'keplerian':
                a : semi-major axis [AU]
                e : eccentricity [degrees]
                i : inclination [degrees]
                Omega : longitude of the ascending node [degrees]
                omega : argument of periapsis [degrees]
                M0 : mean anomaly [degrees]
            If 'cometary':
                p : perihelion distance [AU]
                e : eccentricity [degrees]
                i : inclination [degrees]
                Omega : longitude of the ascending node [degrees]
                omega : argument of periapsis [degrees]
                T0 : time of perihelion passage [degrees]
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
            Time scale of the MJD epochs.
        magnitude : float or `~numpy.ndarray` (N), optional
            Absolute H-magnitude or M1 magnitude.
        slope : float or `~numpy.ndarray` (N), optional
            Photometric slope parameter G or K1.
        observatory_code : str, optional
            Observatory code for which to generate topocentric ephemeris.
        dynamical_model : {'N', '2'}, optional
            Propagate using N or 2-body dynamics.
        ephemeris_file : str, optional
            Which JPL ephemeris file to use with PYOORB.
        """
        # Convert orbits into PYOORB format
        orbits_pyoorb = self._configure_orbits(
            orbits.cartesian,
            orbits.epochs.tt.mjd,
            OpenOrbOrbitType.CARTESIAN,
            OpenOrbTimescale.TT,
            magnitude=None,
            slope=None,
        )

        columns = [
            "mjd_utc",
            "RA_deg",
            "Dec_deg",
            "vRAcosDec",
            "vDec",
            "PhaseAngle_deg",
            "SolarElon_deg",
            "r_au",
            "delta_au",
            "VMag",
            "PosAngle_deg",
            "TLon_deg",
            "TLat_deg",
            "TOCLon_deg",
            "TOCLat_deg",
            "HLon_deg",
            "HLat_deg",
            "HOCLon_deg",
            "HOCLat_deg",
            "Alt_deg",
            "SolarAlt_deg",
            "LunarAlt_deg",
            "LunarPhase",
            "LunarElon_deg",
            "obj_x",
            "obj_y",
            "obj_z",
            "obj_vx",
            "obj_vy",
            "obj_vz",
            "obs_x",
            "obs_y",
            "obs_z",
            "TrueAnom",
        ]

        ephemeris_dfs = []
        for observatory_code, observation_times in observers.items():

            # Convert epochs into PYOORB format
            epochs_pyoorb = self._configure_epochs(
                observation_times.utc.mjd, OpenOrbTimescale.UTC
            )

            # Generate ephemeris
            ephemeris, err = oo.pyoorb.oorb_ephemeris_full(
                in_orbits=orbits_pyoorb,
                in_obscode=observatory_code,
                in_date_ephems=epochs_pyoorb,
                in_dynmodel=self.dynamical_model,
            )

            if err == 1:
                warnings.warn("PYOORB has returned an error!", UserWarning)

            ephemeris = pd.DataFrame(np.vstack(ephemeris), columns=columns)

            ids = np.arange(0, orbits.num_orbits)
            ephemeris["orbit_id"] = [i for i in ids for j in observation_times.utc.mjd]
            ephemeris["observatory_code"] = [
                observatory_code for i in range(len(ephemeris))
            ]
            ephemeris = ephemeris[["orbit_id", "observatory_code"] + columns]

            ephemeris_dfs.append(ephemeris)

        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.sort_values(
            by=["orbit_id", "observatory_code", "mjd_utc"],
            inplace=True,
            ignore_index=True,
        )

        if orbits.ids is not None:
            ephemeris["orbit_id"] = orbits.ids[ephemeris["orbit_id"].values]

        return ephemeris
