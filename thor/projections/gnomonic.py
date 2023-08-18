from typing import Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates import (
    CartesianCoordinates,
    Origin,
    Times,
    coords_from_dataframe,
    coords_to_dataframe,
    transform_covariances_jacobian,
)
from typing_extensions import Self

from . import covariances, transforms


class GnomonicCoordinates(qv.Table):

    time = Times.as_column(nullable=True)
    theta_x = qv.Float64Column()
    theta_y = qv.Float64Column()
    vtheta_x = qv.Float64Column(nullable=True)
    vtheta_y = qv.Float64Column(nullable=True)
    covariance = covariances.ProjectionCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = qv.StringAttribute("testorbit?")

    @property
    def values(self) -> np.ndarray:
        return np.array(
            self.table.select(["theta_x", "theta_y", "vtheta_x", "vtheta_y"])
        ).T

    @property
    def sigma_theta_x(self) -> np.ndarray:
        """
        1-sigma uncertainty in the theta_x coordinate.
        """
        return self.covariance.sigmas[:, 0]

    @property
    def sigma_theta_y(self) -> np.ndarray:
        """
        1-sigma uncertainty in the theta_y coordinate.
        """
        return self.covariance.sigmas[:, 1]

    @property
    def sigma_vtheta_x(self) -> np.ndarray:
        """
        1-sigma uncertainty in the theta_x coordinate velocity.
        """
        return self.covariance.sigmas[:, 2]

    @property
    def sigma_vtheta_y(self):
        """
        1-sigma uncertainty in the theta_y coordinate velocity.
        """
        return self.covariance.sigmas[:, 3]

    @classmethod
    def from_cartesian(
        cls,
        cartesian: CartesianCoordinates,
        center_cartesian: Optional[CartesianCoordinates] = None,
    ) -> Self:
        """
        Create a GnomonicCoordinates object from a CartesianCoordinates object.

        Parameters
        ----------
        cartesian : `~adam_core.coordinates.cartesian.CartesianCoordinates`
            Cartesian coordinates to be transformed.
        center_cartesian : `CartesianCoordinates`, optional
            Cartesian coordinates of the center of the projection. Default is the
            x-axis at 1 au.

        Returns
        -------
        gnomonic : `~thor.projections.gnomonic.GnomonicCoordinates`
            Gnomonic coordinates.
        """
        assert len(cartesian.origin.code.unique()) == 1

        # Check if input coordinates have times defined, if they do lets make
        # sure that we have the correct cartesian center coordinates for each
        if center_cartesian is None and cartesian.time is None:
            # Both center and input coordinates have no times defined, so we
            # can just use the default center coordinates
            center_cartesian = CartesianCoordinates.from_kwargs(
                x=[1],
                y=[0],
                z=[0],
                vx=[0],
                vy=[0],
                vz=[0],
                frame=cartesian.frame,
                origin=cartesian[0].origin,
            )

            # Set the linkage key to be the same integers for each cartesian
            # coordinate and center cartesian coordinate
            left_key = pa.array(np.zeros(len(cartesian)), type=pa.int64())
            right_key = pa.array(np.zeros(len(center_cartesian)), type=pa.int64())

        elif center_cartesian is None and cartesian.time is not None:
            # Create a center cartesian coordinate for each unique time in the
            # input cartesian coordinates
            times = cartesian.time.jd().unique()
            num_unique_times = len(times)
            center_cartesian = CartesianCoordinates.from_kwargs(
                time=cartesian.time.unique(),
                x=np.ones(num_unique_times, dtype=np.float64),
                y=np.zeros(num_unique_times, dtype=np.float64),
                z=np.zeros(num_unique_times, dtype=np.float64),
                vx=np.zeros(num_unique_times, dtype=np.float64),
                vy=np.zeros(num_unique_times, dtype=np.float64),
                vz=np.zeros(num_unique_times, dtype=np.float64),
                frame=cartesian.frame,
                origin=qv.concatenate(
                    [cartesian[0].origin for _ in range(num_unique_times)]
                ),
            )
            left_key = cartesian.time.mjd()
            right_key = center_cartesian.time.mjd()

        elif center_cartesian is not None and cartesian.time is not None:
            assert cartesian.time.scale == center_cartesian.time.scale

            times = cartesian.time.mjd().unique().sort()
            center_times = cartesian.time.mjd().unique().sort()
            if not pc.all(pc.equal(times, center_times)).as_py():
                raise ValueError(
                    "Input cartesian coordinates and projection center cartesian coordinates "
                    "must have the same times."
                )

            left_key = cartesian.time.mjd()
            right_key = center_cartesian.time.mjd()

        else:
            raise ValueError(
                "Input cartesian coordinates and projection center cartesian coordinates "
                "must have the same times."
            )

        # Create a linkage between the cartesian coordinates and the center cartesian
        # coordinates on time
        link = qv.Linkage(
            cartesian,
            center_cartesian,
            left_keys=left_key,
            right_keys=right_key,
        )

        gnomonic_coords = []
        for key, cartesian_i, center_cartesian_i in link.iterate():
            assert len(center_cartesian_i) == 1

            coords_gnomonic, M = transforms.cartesian_to_gnomonic(
                cartesian_i.values,
                center_cartesian=center_cartesian_i.values[0],
            )
            coords_gnomonic = np.array(coords_gnomonic)

            if not cartesian_i.covariance.is_all_nan():
                cartesian_covariances = cartesian_i.covariance.to_matrix()
                covariances_gnomonic = transform_covariances_jacobian(
                    cartesian_i.values,
                    cartesian_covariances,
                    transforms._cartesian_to_gnomonic,
                    in_axes=(0, None),
                    center_cartesian=center_cartesian_i.values[0],
                )
            else:
                covariances_gnomonic = np.empty(
                    (len(coords_gnomonic), 4, 4), dtype=np.float64
                )
                covariances_gnomonic.fill(np.nan)

            gnomonic_coords.append(
                cls.from_kwargs(
                    theta_x=coords_gnomonic[:, 0],
                    theta_y=coords_gnomonic[:, 1],
                    vtheta_x=coords_gnomonic[:, 2],
                    vtheta_y=coords_gnomonic[:, 3],
                    time=cartesian_i.time,
                    covariance=covariances.ProjectionCovariances.from_matrix(
                        covariances_gnomonic
                    ),
                    origin=cartesian_i.origin,
                    frame=cartesian_i.frame,
                )
            )

        return qv.concatenate(gnomonic_coords)

    def to_dataframe(
        self, sigmas: bool = False, covariances: bool = True
    ) -> pd.DataFrame:
        """
        Convert coordinates to a pandas DataFrame.

        Parameters
        ----------
        sigmas : bool, optional
            If True, include 1-sigma uncertainties in the DataFrame.
        covariances : bool, optional
            If True, include covariance matrices in the DataFrame. Covariance matrices
            will be split into 21 columns, with the lower triangular elements stored.

        Returns
        -------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        """
        return coords_to_dataframe(
            self,
            ["theta_x", "theta_y", "vtheta_x", "vtheta_y"],
            sigmas=sigmas,
            covariances=covariances,
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, frame: Literal["ecliptic", "equatorial"]
    ) -> Self:
        """
        Create coordinates from a pandas DataFrame.

        Parameters
        ----------
        df : `~pandas.Dataframe`
            DataFrame containing coordinates.
        frame : {"ecliptic", "equatorial"}
            Frame in which coordinates are defined.

        Returns
        -------
        coords : `~thor.projections.gnomonic.GnomonicCoordinates`
            Gnomomic projection coordinates.
        """
        return coords_from_dataframe(
            cls,
            df,
            coord_names=["theta_x", "theta_y", "vtheta_x", "vtheta_y"],
            frame=frame,
        )
