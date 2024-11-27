import healpy as hp
import numpy as np
import numpy.typing as npt


def calculate_healpixels(
    ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64], nside: int
) -> npt.NDArray[np.int64]:
    """
    Calculate the healpixel for a set of RA and Dec values. Returned healpixels are in the nested
    ordering scheme.

    Parameters
    ----------
    ra
        Right ascension values in degrees.
    dec
        Declination values in degrees.

    Returns
    -------
    healpixels
        Healpixels in the nested ordering scheme.
    """
    return hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
