import numpy as np
from astropy.time import Time

__all__ = [
    "_unpackMPCDate",
    "_lookupMPC",
    "convertMPCPackedDates"
]


def _unpackMPCDate(epoch_pf):
    # Taken from Lynne Jones' SSO TOOLS. 
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    epoch_pf = str(epoch_pf)
    year = _lookupMPC(epoch_pf[0])*100 + int(epoch_pf[1:3])
    month = _lookupMPC(epoch_pf[3])
    day = _lookupMPC(epoch_pf[4])
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)
    
    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60) 
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)
        
    return isot_string

def _lookupMPC(x):
    # Convert the single character dates into integers.
    try:
        x = int(x)
    except ValueError:
        x = ord(x) - 55
    if x < 0 or x > 31:
        raise ValueError
    return x

def convertMPCPackedDates(pf_tt):
    """
    Convert MPC packed form dates (in the TT time scale) to 
    MJDs in TT. See: https://minorplanetcenter.net/iau/info/PackedDates.html
    for details on the packed date format.

    Parameters
    ----------
    pf_tt : `~numpy.ndarray` (N)
        MPC-style packed form epochs in the TT time scale.

    Returns
    -------
    mjd_tt : `~numpy.ndarray` (N)
        Epochs in TT MJDs.
    """
    pf_tt = np.asarray(pf_tt)
    isot_tt = np.empty(len(pf_tt), dtype="<U32")
    
    for i, epoch in enumerate(pf_tt):
        isot_tt[i] = _unpackMPCDate(epoch)
    
    epoch = Time(isot_tt, format="isot", scale="tt")
    return epoch.tt.mjd