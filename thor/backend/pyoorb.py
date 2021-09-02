import os
import warnings
import numpy as np
import pyoorb as oo
import pandas as pd
from astropy.time import Time

from ..utils import _checkTime
from .backend import Backend

PYOORB_CONFIG = {
    "dynamical_model" : "N",
    "ephemeris_file" : "de430.dat"
}

class PYOORB(Backend):

    def __init__(self, **kwargs):
        # Make sure only the correct kwargs
        # are passed to the constructor
        allowed_kwargs = PYOORB_CONFIG.keys()
        for k in kwargs:
            if k not in allowed_kwargs:
                raise ValueError()

        # If an allowed kwarg is missing, add the
        # default
        for k in allowed_kwargs:
            if k not in kwargs:
                kwargs[k] = PYOORB_CONFIG[k]

        super().__init__(name="OpenOrb", **kwargs)

        self.setup()
        return

    def setup(self):
        """
        Initialize PYOORB with the designated JPL ephemeris file.

        """
        env_var = f"THOR_PYOORB"
        if env_var in os.environ.keys() and os.environ[env_var] == "True":
            pass
        else:
            if os.environ.get("OORB_DATA") == None:
                os.environ["OORB_DATA"] = os.path.join(os.environ["CONDA_PREFIX"], "share/openorb")
            # Prepare pyoorb
            ephfile = os.path.join(os.getenv('OORB_DATA'), self.ephemeris_file)
            err = oo.pyoorb.oorb_init(ephfile)
            if err == 0:
                os.environ[env_var] = "True"
                self.__env_var = env_var
                self.is_setup = True
            else:
                warnings.warn("PYOORB returned error code: {}".format(err))

        return

    def _configureOrbits(self, orbits, t0, orbit_type, time_scale, magnitude, slope):
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        orbits : `~numpy.ndarray` (N, 6)
            Orbits to convert. See orbit_type for expected input format.
        t0 : `~numpy.ndarray` (N)
            Epoch in MJD at which the orbits are defined.
        orbit_type : {'cartesian', 'keplerian', 'cometary'}, optional
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
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}, optional
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
        if orbits_.shape == (6,):
            num_orbits = 1
        else:
            num_orbits = orbits_.shape[0]

        if orbit_type == "cartesian":
            orbit_type = [1 for i in range(num_orbits)]
        elif orbit_type == "cometary":
            orbit_type = [2 for i in range(num_orbits)]
            H = M1
            G = K1
            orbits_[:, 1:5] = np.radians(orbits_[:, 1:5])
        elif orbit_type == "keplerian":
            orbit_type = [3 for i in range(num_orbits)]
            orbits_[:, 1:] = np.radians(orbits_[:, 1:])
        else:
            raise ValueError("orbit_type should be one of {'cartesian', 'keplerian', 'cometary'}")

        if time_scale == "UTC":
            time_scale = [1 for i in range(num_orbits)]
        elif time_scale == "UT1":
            time_scale = [2 for i in range(num_orbits)]
        elif time_scale == "TT":
            time_scale = [3 for i in range(num_orbits)]
        elif time_scale == "TAI":
            time_scale = [4 for i in range(num_orbits)]
        else:
            raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

        if slope is not None:
            if not isinstance(slope, np.ndarray):
                slope = np.array([slope for i in range(num_orbits)])
        else:
            slope = [0.15 for i in range(num_orbits)]

        if magnitude is not None:
            if not isinstance(magnitude, np.ndarray):
                magnitude = np.array([magnitude for i in range(num_orbits)])
        else:
            magnitude = [20.0 for i in range(num_orbits)]

        ids = [i for i in range(num_orbits)]

        if num_orbits > 1:
            orbits_pyoorb = np.array(
                np.array([
                    ids,
                    *list(orbits_.T),
                     orbit_type,
                     t0,
                     time_scale,
                     magnitude,
                     slope
                ]).T,
                dtype=np.double,
                order='F'
            )
        else:
            orbits_pyoorb = np.array([
                [
                    ids[0],
                    *list(orbits_.T),
                    orbit_type[0],
                    t0[0],
                    time_scale[0],
                    magnitude[0],
                    slope[0]]
                ],
                dtype=np.double,
                order='F'
            )

        return orbits_pyoorb

    def _configureEpochs(self, epochs, time_scale):
        """
        Convert an array of orbits into the format expected by PYOORB.

        Parameters
        ----------
        epochs : `~numpy.ndarray` (N)
            Epoch in MJD to convert.
        time_scale : {'UTC', 'UT1', 'TT', 'TAI'}
            Time scale of the MJD epochs.

        Returns
        -------
        epochs_pyoorb : `~numpy.ndarray (N, 2)
            Epochs converted into the PYOORB format.
        """
        num_times = len(epochs)
        if time_scale == "UTC":
            time_scale = [1 for i in range(num_times)]
        elif time_scale == "UT1":
            time_scale = [2 for i in range(num_times)]
        elif time_scale == "TT":
            time_scale = [3 for i in range(num_times)]
        elif time_scale == "TAI":
            time_scale = [4 for i in range(num_times)]
        else:
            raise ValueError("time_scale should be one of {'UTC', 'UT1', 'TT', 'TAI'}")

        epochs_pyoorb = np.array(list(np.vstack([epochs, time_scale]).T), dtype=np.double, order='F')
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
        orbits_pyoorb = self._configureOrbits(
            orbits.cartesian,
            orbits.epochs.tt.mjd,
            "cartesian",
            "TT",
            orbits.H,
            orbits.G
        )

        # Convert epochs into PYOORB format
        epochs_pyoorb = self._configureEpochs(t1.tt.mjd, "TT")

        # Propagate orbits to each epoch and append to list
        # of new states
        states = []
        orbits_pyoorb_i = orbits_pyoorb.copy()
        for epoch in epochs_pyoorb:
            orbits_pyoorb_i, err = oo.pyoorb.oorb_propagation(
                in_orbits=orbits_pyoorb_i,
                in_epoch=epoch,
                in_dynmodel=self.dynamical_model
            )
            states.append(orbits_pyoorb_i)

        # Convert list of new states into a pandas data frame
        # These states at the moment will always be return as cartesian
        # state vectors
        elements = ["x", "y", "z", "vx", "vy", "vz"]
        # Other PYOORB state vector representations:
        #"keplerian":
        #    elements = ["a", "e", "i", "Omega", "omega", "M0"]
        #"cometary":
        #    elements = ["q", "e", "i", "Omega", "omega", "T0"]

        # Create pandas data frame
        columns = [
            "orbit_id",
            *elements,
            "orbit_type",
            "epoch_mjd",
            "time_scale",
            "H/M1",
            "G/K1"
        ]
        propagated = pd.DataFrame(
            np.concatenate(states),
            columns=columns
        )
        propagated["orbit_id"] = propagated["orbit_id"].astype(int)

        # Convert output epochs to TDB
        epochs = Time(
            propagated["epoch_mjd"].values,
            format="mjd",
            scale="tt"
        )
        propagated["mjd_tdb"] = epochs.tdb.value

        # Drop PYOORB specific columns (may want to consider this option later on.)
        propagated.drop(
            columns=[
                "epoch_mjd",
                "orbit_type",
                "time_scale",
                "H/M1",
                "G/K1"
            ],
            inplace=True
        )

        # Re-order columns and sort
        propagated = propagated[["orbit_id", "mjd_tdb"] + elements]
        propagated.sort_values(
            by=["orbit_id", "mjd_tdb"],
            inplace=True,
            ignore_index=True
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
        orbits_pyoorb = self._configureOrbits(
            orbits.cartesian,
            orbits.epochs.tt.mjd,
            "cartesian",
            "TT",
            orbits.H,
            orbits.G
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
            "TrueAnom"
        ]

        ephemeris_dfs = []
        for observatory_code, observation_times in observers.items():
            _checkTime(observation_times, "observation_times")

            # Convert epochs into PYOORB format
            epochs_pyoorb = self._configureEpochs(observation_times.utc.mjd, "UTC")

            # Generate ephemeris
            ephemeris, err = oo.pyoorb.oorb_ephemeris_full(
              in_orbits=orbits_pyoorb,
              in_obscode=observatory_code,
              in_date_ephems=epochs_pyoorb,
              in_dynmodel=self.dynamical_model
            )

            if err == 1:
                warnings.warn("PYOORB has returned an error!", UserWarning)

            ephemeris = pd.DataFrame(
                np.vstack(ephemeris),
                columns=columns
            )

            ids = np.arange(0, orbits.num_orbits)
            ephemeris["orbit_id"] = [i for i in ids for j in observation_times.utc.mjd]
            ephemeris["observatory_code"] = [observatory_code for i in range(len(ephemeris))]
            ephemeris = ephemeris[["orbit_id", "observatory_code"] + columns]

            ephemeris_dfs.append(ephemeris)

        ephemeris = pd.concat(ephemeris_dfs)
        ephemeris.sort_values(
            by=["orbit_id", "observatory_code", "mjd_utc"],
            inplace=True,
            ignore_index=True
        )

        if orbits.ids is not None:
            ephemeris["orbit_id"] = orbits.ids[ephemeris["orbit_id"].values]

        return ephemeris
