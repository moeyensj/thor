
import numpy as np

__all___ = [
    "convert_SBDB_covariances"
]

def convert_SBDB_covariances(sbdb_covariances):
    """
    Convert SBDB covariance matrices to Cometary covariance matrices.

    Parameters
    ----------
    sbdb_covariances : `~numpy.ndarray` (N, 6, 6)
        Covariance matrices pulled from JPL's Small Body Database Browser.

    Returns
    -------
    covariances : `~numpy.ndarray` (N, 6, 6)
        Cometary covariance matrices.
    """
    covariances = np.zeros_like(sbdb_covariances)
    # sigma_q{x}
    covariances[:, 0, 0] = sbdb_covariances[:, 1, 1] # sigma_qq
    covariances[:, 1, 0] = covariances[:, 0, 1] = sbdb_covariances[:, 0, 1] # sigma_qe
    covariances[:, 2, 0] = covariances[:, 0, 2] = sbdb_covariances[:, 5, 1] # sigma_qi
    covariances[:, 3, 0] = covariances[:, 0, 3] = sbdb_covariances[:, 3, 1] # sigma_qraan
    covariances[:, 4, 0] = covariances[:, 0, 4] = sbdb_covariances[:, 4, 1] # sigma_qap
    covariances[:, 5, 0] = covariances[:, 0, 5] = sbdb_covariances[:, 2, 1] # sigma_qtp

    # sigma_e{x}
    covariances[:, 1, 1] = sbdb_covariances[:, 0, 0] # sigma_ee
    covariances[:, 2, 1] = covariances[:, 1, 2] = sbdb_covariances[:, 5, 0] # sigma_ei
    covariances[:, 3, 1] = covariances[:, 1, 3] = sbdb_covariances[:, 3, 0] # sigma_eraan
    covariances[:, 4, 1] = covariances[:, 1, 4] = sbdb_covariances[:, 4, 0] # sigma_eap
    covariances[:, 5, 1] = covariances[:, 1, 5] = sbdb_covariances[:, 2, 0] # sigma_etp

    # sigma_i{x}
    covariances[:, 2, 2] = sbdb_covariances[:, 5, 5] # sigma_ii
    covariances[:, 3, 2] = covariances[:, 2, 3] = sbdb_covariances[:, 3, 5] # sigma_iraan
    covariances[:, 4, 2] = covariances[:, 2, 4] = sbdb_covariances[:, 4, 5] # sigma_iap
    covariances[:, 5, 2] = covariances[:, 2, 5] = sbdb_covariances[:, 2, 5] # sigma_itp

    # sigma_raan{x}
    covariances[:, 3, 3] = sbdb_covariances[:, 3, 3] # sigma_raanraan
    covariances[:, 4, 3] = covariances[:, 3, 4] = sbdb_covariances[:, 4, 3] # sigma_raanap
    covariances[:, 5, 3] = covariances[:, 3, 5] = sbdb_covariances[:, 2, 3] # sigma_raantp

    # sigma_ap{x}
    covariances[:, 4, 4] = sbdb_covariances[:, 4, 4] # sigma_apap
    covariances[:, 5, 4] = covariances[:, 4, 5] = sbdb_covariances[:, 2, 4] # sigma_aptp

    # sigma_tp{x}
    covariances[:, 5, 5] = sbdb_covariances[:, 2, 2] # sigma_tptp


    return covariances