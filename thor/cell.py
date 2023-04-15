import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = ["Cell"]


class Cell:
    """
    Cell: Construct used to find detections around a test orbit.

    Parameters
    ----------
    center : `~numpy.ndarray` (2)
        Location of the cell's center (typically the sky-plane location of the test orbit).
    mjd_utc : float
        Time in units of MJD (must be MJD scale).
    area : float, optional
        Cell's area in units of squared degrees.
        [Default = 10]

    Returns
    -------
    None
    """

    def __init__(self, center, mjd_utc, area=10):
        self.center = center
        self.mjd_utc = mjd_utc
        self.observations = None
        self.area = area
        return

    def getObservations(self, observations):
        """
        Get the observations that lie within the cell. Populates
        self.observations with the observations from self.dataframe
        that lie within the cell.

        Note: Looks for observations with +- 0.00001 MJD of self.mjd

        Parameters
        ----------
        observations : `~pandas.DataFrame`
            Data frame containing observations with at minimum the following columns:
            exposure time in MJD, observation ID, RA in degrees, Dec in degrees, the heliocentric ecliptic location of
            the observer in AU.

        Returns
        -------
        None
        """
        exp_observations = observations[
            (observations["mjd_utc"] <= self.mjd_utc + 0.00001)
            & (observations["mjd_utc"] >= self.mjd_utc - 0.00001)
        ]
        obs_ids = exp_observations["obs_id"].values

        coords_observations = SkyCoord(
            *exp_observations[["RA_deg", "Dec_deg"]].values.T,
            unit=u.degree,
            frame="icrs"
        )
        coords_center = SkyCoord(*self.center, unit=u.degree, frame="icrs")

        # Find all coordinates within circular region centered about the center coordinate
        distance = coords_center.separation(coords_observations).degree
        keep = obs_ids[np.where(distance <= np.sqrt(self.area / np.pi))[0]]

        self.observations = exp_observations[
            exp_observations["obs_id"].isin(keep)
        ].copy()
        return
