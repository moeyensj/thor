import pytest
import numpy as np
import numpy.testing as npt

from ..sbdb import convert_SBDB_covariances

def test_convert_SBDB_covariances():

    sbdb_format = np.array([
        ["e_e", "e_q", "e_tp", "e_raan", "e_ap", "e_i"],
        ["q_e", "q_q", "q_tp", "q_raan", "q_ap", "q_i"],
        ["tp_e", "tp_q", "tp_tp", "tp_raan", "tp_ap", "tp_i"],
        ["raan_e", "raan_q", "raan_tp", "raan_raan", "raan_ap", "raan_i"],
        ["ap_e", "ap_q", "ap_tp", "ap_raan", "ap_ap", "ap_i"],
        ["i_e", "i_q", "i_tp", "i_raan", "i_ap", "i_i"],
    ])
    thor_format = np.array([
        ["q_q", "q_e", "q_i", "q_raan", "q_ap", "q_tp"],
        ["e_q", "e_e", "e_i", "e_raan", "e_ap", "e_tp"],
        ["i_q", "i_e", "i_i", "i_raan", "i_ap", "i_tp"],
        ["raan_q", "raan_e", "raan_i", "raan_raan", "raan_ap", "raan_tp"],
        ["ap_q", "ap_e", "ap_i", "ap_raan", "ap_ap", "ap_tp"],
        ["tp_q", "tp_e", "tp_i", "tp_raan", "tp_ap", "tp_tp"],
    ])
    sbdb_format = np.array([sbdb_format])
    thor_format = np.array([thor_format])

    cometary_covariances = convert_SBDB_covariances(sbdb_format)
    # Some of the symmetric terms will be in reverse format: for example
    # q_e will be e_q where it should be q_e. So for places where
    # the order is reversed, let's try to flip them around and then lets
    # check for equality with the expected THOR format.

    flip_mask = np.where(cometary_covariances != thor_format)
    for i, j, k in zip(*flip_mask):
        cometary_covariances[i, j, k] = "_".join(cometary_covariances[i, j, k].split("_")[::-1])

    npt.assert_equal(cometary_covariances, thor_format)


