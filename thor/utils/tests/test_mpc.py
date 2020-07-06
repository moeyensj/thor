import numpy as np
from astropy.time import Time

from ..mpc import convertMPCPackedDates
        
def test_convertMPCPackedDates():

    # Use a few modified examples from https://minorplanetcenter.net/iau/info/PackedDates.html
    # and test conversion from packed form to MJDs 
    isot_tt = np.array([
        "1996-01-01",   
        "1996-01-10",   
        "1996-09-30",   
        "1996-10-01",
        "2001-10-22",    
        "2001-10-22T00:00:00.0000",
        "2001-10-22T12:00:00.0000",
        "1996-09-30T18:00:00.0000",
        "1996-09-30T18:45:00.0000",
    ])

    pf_tt = np.array([
        "J9611", 
        "J961A", 
        "J969U", 
        "J96A1", 
        "K01AM",
        "K01AM",
        "K01AM5",
        "J969U75",
        "J969U78125"
    ])
    
    mjd_tt = convertMPCPackedDates(pf_tt)
    mjd = Time(isot_tt, format="isot", scale="tt")

    np.testing.assert_equal(mjd_tt, mjd.tt.mjd)
    return