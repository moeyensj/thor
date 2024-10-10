from typing import Optional

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import CartesianCoordinates, Origin
from adam_core.coordinates.covariances import transform_covariances_jacobian
from adam_core.time import Timestamp
from typing_extensions import Self

from .covariances import ProjectionCovariances
from .transforms import _cartesian_to_gnomonic, cartesian_to_gnomonic


class GnomonicCoordinates(qv.Table):

    time = Timestamp.as_column(nullable=True)
    theta_x = qv.Float64Column()
    theta_y = qv.Float64Column()
    vtheta_x = qv.Float64Column(nullable=True)
    vtheta_y = qv.Float64Column(nullable=True)
    covariance = ProjectionCovariances.as_column(nullable=True)
    origin = Origin.as_column()
    frame = qv.StringAttribute("testorbit?")

    @property
    def values(self) -> np.ndarray:
        return np.array(self.table.select(["theta_x", "theta_y", "vtheta_x", "vtheta_y"]))

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

            # Create a linkage between the cartesian coordinates and the center cartesian
            # coordinates on time
            link = qv.Linkage(
                cartesian,
                center_cartesian,
                left_keys=left_key,
                right_keys=right_key,
            )

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
                origin=qv.concatenate([cartesian[0].origin for _ in range(num_unique_times)]),
            )
            link = cartesian.time.link(center_cartesian.time, precision="ms")

        elif center_cartesian is not None and cartesian.time is not None:
            assert cartesian.time.scale == center_cartesian.time.scale

            link = cartesian.time.link(center_cartesian.time, precision="ms")

        else:
            raise ValueError(
                "Input cartesian coordinates and projection center cartesian coordinates "
                "must have the same times."
            )

        # Round the times to the nearest microsecond and use those to select
        # the cartesian coordinates and center cartesian coordinates
        rounded_cartesian_times = cartesian.time.rounded(precision="ms")  # type: ignore
        rounded_center_cartesian_times = center_cartesian.time.rounded(precision="ms")  # type: ignore

        gnomonic_coords = GnomonicCoordinates.empty()
        for key, time_i, center_time_i in link.iterate():

            cartesian_i = cartesian.apply_mask(
                rounded_cartesian_times.equals_scalar(key[0], key[1], precision="ms")
            )
            center_cartesian_i = center_cartesian.apply_mask(
                rounded_center_cartesian_times.equals_scalar(key[0], key[1], precision="ms")
            )

            if len(center_cartesian_i) == 0:
                raise ValueError("No center cartesian coordinates found for this time.")

            coords_gnomonic, M = cartesian_to_gnomonic(
                cartesian_i.values,
                center_cartesian=center_cartesian_i.values[0],
            )
            coords_gnomonic = np.array(coords_gnomonic)

            if not cartesian_i.covariance.is_all_nan():
                cartesian_covariances = cartesian_i.covariance.to_matrix()
                covariances_gnomonic = transform_covariances_jacobian(
                    cartesian_i.values,
                    cartesian_covariances,
                    _cartesian_to_gnomonic,
                    in_axes=(0, None),
                    center_cartesian=center_cartesian_i.values[0],
                )
            else:
                covariances_gnomonic = np.empty((len(coords_gnomonic), 4, 4), dtype=np.float64)
                covariances_gnomonic.fill(np.nan)

            gnomonic_coords_chunk = cls.from_kwargs(
                theta_x=coords_gnomonic[:, 0],
                theta_y=coords_gnomonic[:, 1],
                vtheta_x=coords_gnomonic[:, 2],
                vtheta_y=coords_gnomonic[:, 3],
                time=time_i,
                covariance=ProjectionCovariances.from_matrix(covariances_gnomonic),
                origin=cartesian_i.origin,
                frame=cartesian_i.frame,
            )

            gnomonic_coords = qv.concatenate([gnomonic_coords, gnomonic_coords_chunk])
            if gnomonic_coords.fragmented():
                gnomonic_coords = qv.defragment(gnomonic_coords)

        return gnomonic_coords
