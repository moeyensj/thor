from typing import Tuple

import healpy as hp
import numpy as np
import numpy.typing as npt

from .healpixel import compute_lon_lat_boundaries


def split_phase_space(
    mu: np.ndarray,
    cov: np.ndarray,
    dt: float,
    k: int = 3,
    beta: float = 0.6,
    gamma: float = 0.9,
    num: int = 2,
    current_depth: int = 0,
    max_depth: int = 0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Recursively split phase space and return stacked states and covariances.

    Parameters
    ----------
    mu : (6,) numpy.ndarray
        State vector.
    cov : (6, 6) numpy.ndarray
        State covariance matrix.
    dt : float
        Time step used to scale position/velocity coupling. Cursory experiment suggests that this should be the same
        length as the linking window.
    k : int, optional
        Spread factor for child offsets along principal axis.
    beta : float, optional
        Shrink factor applied to leading eigenvalue for children.
    gamma : float, optional
        Shrink factor applied to non-leading eigenvalues for children.
    num : int, optional
        Number of children to create per split.
    current_depth : int, optional
        Current recursion depth.
    max_depth : int, optional
        Maximum recursion depth (inclusive).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - states with shape (N, 6)
        - covariances with shape (N, 6, 6)
    """
    if current_depth > max_depth:
        return mu.reshape(1, 6), cov.reshape(1, 6, 6)

    S = np.zeros((6, 6))
    S[0:3, 0:3] = np.eye(3)
    S[3:6, 3:6] = np.eye(3) * dt
    S_inv = np.linalg.inv(S)

    mu_transformed = S @ mu
    cov_transformed = S @ cov @ S.T

    eigenvalues, eigenvectors = np.linalg.eig(cov_transformed)

    lambd1 = eigenvalues[0]
    vec1 = eigenvectors[:, 0]

    max_offset = k * (1 - beta)

    if num == 1:
        children_offsets = np.array([0.0])
    else:
        children_offsets = np.linspace(-max_offset, max_offset, num)

    deltas = np.sqrt(lambd1) * vec1 * children_offsets.reshape(-1, 1)
    children_transformed = mu_transformed.reshape(1, 6) + deltas

    children_eigenvalues = eigenvalues.copy()
    children_eigenvalues[0] = (beta**2) * lambd1
    children_eigenvalues[1:] = (gamma**2) * children_eigenvalues[1:]
    children_cov_transformed = eigenvectors @ np.diag(children_eigenvalues) @ eigenvectors.T

    states_list = []
    covs_list = []

    for child_transformed in children_transformed:
        child_cov_t = children_cov_transformed.copy()
        child = S_inv @ child_transformed
        child_cov = S_inv.T @ child_cov_t @ S_inv

        child_states, child_covs = split_phase_space(
            child,
            child_cov,
            dt,
            k=k,
            beta=beta,
            gamma=gamma,
            num=num,
            current_depth=current_depth + 1,
            max_depth=max_depth,
        )
        states_list.append(child_states)
        covs_list.append(child_covs)

    states = np.concatenate(states_list, axis=0)
    covs = np.concatenate(covs_list, axis=0)
    return states, covs


def split_healpixel(
    mu: np.ndarray,
    cov: np.ndarray,
    current_nside: int,
    new_nside: int,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Split a spherical state and covariance into child HEALPix pixels, refining only
    lon/lat and their variances to the child-pixel centers and half-widths.
    Other state components and covariances are preserved (except lon/lat cross-terms set to zero).

    Parameters
    ----------
    mu : (6,) numpy.ndarray
        Spherical state vector ordered as [rho, lon, lat, vrho, vlon, vlat] (degrees for angles).
    cov : (6, 6) numpy.ndarray
        Covariance matrix corresponding to mu.
    current_nside : int
        NSIDE of the parent HEALPix pixel (NESTED indexing).
    new_nside : int
        NSIDE to refine to. Must be a power-of-two multiple of current_nside.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - states with shape (N_children, 6) in spherical order [rho, lon, lat, vrho, vlon, vlat]
        - covariances with shape (N_children, 6, 6)
    """
    if new_nside <= current_nside:
        # No refinement or invalid request; return original
        return mu.reshape(1, 6), cov.reshape(1, 6, 6)

    if (new_nside % current_nside) != 0:
        raise ValueError(f"new_nside ({new_nside}) must be a multiple of current_nside ({current_nside}).")

    factor = new_nside // current_nside
    # factor must be power-of-two for strict NEST subdivision
    if factor & (factor - 1) != 0:
        raise ValueError("new_nside/current_nside must be a power-of-two factor for NESTED subdivision.")

    num_children = factor * factor

    # Parent pixel containing (lon, lat)
    lon0 = mu[1]
    lat0 = mu[2]
    parent_pix = hp.ang2pix(current_nside, lon0, lat0, nest=True, lonlat=True)

    # Child pixels are a contiguous block in NEST indexing
    start = parent_pix * num_children
    children = np.arange(start, start + num_children, dtype=int)

    # Initialize output states/covariances by replicating
    states = np.repeat(mu.reshape(1, 6), num_children, axis=0)
    covs = np.repeat(cov.reshape(1, 6, 6), num_children, axis=0)

    # Fill lon/lat centers and adjust variances
    for i, pix in enumerate(children):
        clon, clat = hp.pix2ang(new_nside, pix, nest=True, lonlat=True)
        lon_b, lat_b = compute_lon_lat_boundaries(new_nside, pix)
        dlon = np.max(np.abs(lon_b - clon))
        dlat = np.max(np.abs(lat_b - clat))

        states[i, 1] = clon
        states[i, 2] = clat

        covs[i, 1, 1] = dlon**2
        covs[i, 2, 2] = dlat**2
        covs[i, 1, 2] = 0.0
        covs[i, 2, 1] = 0.0

    return states, covs
