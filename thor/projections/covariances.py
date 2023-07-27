from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.coordinates.covariances import (
    covariances_from_df,
    covariances_to_df,
    sigmas_from_df,
    sigmas_to_covariances,
    sigmas_to_df,
)
from typing_extensions import Self


class ProjectionCovariances(qv.Table):
    # Based on adam_core's CoordinateCovariances class
    # TODO: Would be interesting if the dimensionality can be generalized
    #      to D dimensions, so (N, D, D) instead of (N, 4, 4). We would be
    #      able to use this class for the covariance matrices of different
    #      measurments like projections (D = 4) and photometry (D = 1).

    # This is temporary while we await the implementation of
    # https://github.com/apache/arrow/issues/35599
    # See more details here:
    values = qv.FixedSizeListColumn(pa.float64(), list_size=16, nullable=True)
    # When fixed, we should revert to:
    # values = Column(pa.fixed_shape_tensor(pa.float64(), (4, 4)))

    @property
    def sigmas(self):
        cov_diag = np.diagonal(self.to_matrix(), axis1=1, axis2=2)
        sigmas = np.sqrt(cov_diag)
        return sigmas

    def to_matrix(self) -> np.ndarray:
        """
        Return the covariance matrices as a 3D array of shape (N, 4, 4).

        Returns
        -------
        covariances : `numpy.ndarray` (N, 4, 4)
            Covariance matrices for N coordinates in 4 dimensions.
        """
        # return self.values.combine_chunks().to_numpy_ndarray()
        values = self.values.to_numpy(zero_copy_only=False)

        # If all covariance matrices are None, then return a covariances
        # filled with NaNs.
        if np.all(values == None):  # noqa: E711
            return np.full((len(self), 4, 4), np.nan)

        else:
            # Try to stack the values into a 3D array. If this works, then
            # all covariance matrices are the same size and we can return
            # the stacked matrices.
            try:
                cov = np.stack(values).reshape(-1, 4, 4)

            # If not then some of the arrays might be None. Lets loop through
            # the values and fill in the arrays that are missing (None) with NaNs.
            except ValueError as e:
                # If we don't get the error we expect, then raise it.
                if str(e) != "all input arrays must have the same shape":
                    raise e
                else:
                    for i in range(len(values)):
                        if values[i] is None:
                            values[i] = np.full(16, np.nan)

                # Try stacking again
                cov = np.stack(values).reshape(-1, 4, 4)

            return cov

    @classmethod
    def from_matrix(cls, covariances: np.ndarray) -> Self:
        """
        Create a Covariances object from a 3D array of covariance matrices.

        Parameters
        ----------
        covariances : `numpy.ndarray` (N, 4, 4)
            Covariance matrices for N coordinates in 4 dimensions.

        Returns
        -------
        covariances : `Covariances`
            Covariance matrices for N coordinates in 4 dimensions.
        """
        # cov = pa.FixedShapeTensorArray.from_numpy_ndarray(covariances)
        cov = covariances.reshape(-1, 16)
        return cls.from_kwargs(values=list(cov))

    @classmethod
    def from_sigmas(cls, sigmas: np.ndarray) -> Self:
        """
        Create a Covariances object from a 2D array of sigmas.

        Parameters
        ----------
        sigmas : `numpy.ndarray` (N, 6)
            Array of 1-sigma uncertainties for N coordinates in 6
            dimensions.

        Returns
        -------
        covariances : `Covariances`
            Covariance matrices with the diagonal elements set to the
            squares of the input sigmas.
        """
        return cls.from_matrix(sigmas_to_covariances(sigmas))

    def to_dataframe(
        self,
        coord_names: List[str] = ["theta_x", "theta_y", "vtheta_x", "vtheta_y"],
        sigmas: bool = False,
    ) -> pd.DataFrame:
        """
        Return the covariance matrices represented as lower triangular columns in a pandas DataFrame.

        Parameters
        ----------
        coord_names : `list` of `str`, optional
            Names of the coordinate axes. Default is ["theta_x", "theta_y", "vtheta_x", "vtheta_y"].
        sigmas : `bool`, optional
            If True, the standard deviations are added as columns to the DataFrame. Default is False.

        Returns
        -------
        df : `pandas.DataFrame`
            Covariance matrices (lower triangular) for N coordinates in 4 dimensions.
        """
        df = covariances_to_df(self.to_matrix(), coord_names=coord_names, kind="lower")
        if sigmas:
            df_sigmas = sigmas_to_df(self.sigmas, coord_names=coord_names)
            df = df_sigmas.join(df)

        return df

    @classmethod
    def from_dataframe(
        cls, df, coord_names: List[str] = ["theta_x", "theta_y", "vtheta_x", "vtheta_y"]
    ) -> Self:
        """
        Create a Covariances object from a pandas DataFrame.

        Parameters
        ----------
        df : `pandas.DataFrame`
            Covariance matrices (lower triangular) for N coordinates in 4 dimensions.
        coord_names : `list` of `str`, optional
            Names of the coordinate axes. Default is ["theta_x", "theta_y", "vtheta_x", "vtheta_y"].

        Returns
        -------
        covariances : `CoordinateCovariances`
            Covariance matrices for N coordinates in 4 dimensions.
        """
        try:
            covariances = covariances_from_df(df, coord_names=coord_names, kind="lower")
        except KeyError:
            sigmas = sigmas_from_df(df, coord_names=coord_names)
            covariances = sigmas_to_covariances(sigmas)

        if np.all(np.isnan(covariances)):
            return cls.from_kwargs(values=[None for i in range(len(covariances))])
        else:
            return cls.from_matrix(covariances)

    def is_all_nan(self) -> bool:
        """
        Check if all covariance matrix values are NaN.

        Returns
        -------
        is_all_nan : bool
            True if all covariance matrix elements are NaN, False otherwise.
        """
        return np.all(np.isnan(self.to_matrix()))
