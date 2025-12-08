from typing import Tuple

import numpy as np
import numpy.typing as npt


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
