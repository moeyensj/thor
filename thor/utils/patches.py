import numpy as np
import healpy as hp

__all__ = [
    "assignPatchesSquare",
    "assignPatchesHEALPix"
]

def assignPatchesSquare(
        ra: np.ndarray,
        dec: np.ndarray,
        ra_width: float = 15.,
        dec_width: float = 15.,
    ) -> np.ndarray:
    """
    Assign a patch ID to each coordinate where a patch is a square region
    on the sky plane of ra_width in RA and of dec_width in Dec.

    RA must be between 0 and 360 degrees.
    Dec must be between -90 and 90 degrees.

    Patches are created first in increasing Declination and second in increasing
    Right Ascension. The first patch will border RA of 0 and Dec of -90, while
    the last patch will border RA of 360 and Dec of 90.

    Parameters
    ----------
    ra : `~numpy.ndarray` (N)
        Right Ascension in degrees.
    dec : `~numpy.ndarray` (N)
        Declination in degrees.
    ra_width : float
        Width of patch in RA in degrees.
    dec_with : float
        Width of patch in Dec in degrees.

    Returns
    -------
    patch_ids : `~numpy.ndarray` (N)
        The patch ID for each coordinate.
    """
    ras = np.arange(0, 360 + ra_width, ra_width)
    decs = np.arange(-90, 90 + dec_width, dec_width)

    patch_id = 0
    patch_ids = -1 * np.ones(len(ra), dtype=int)
    for ra_i, ra_f in zip(ras[:-1], ras[1:]):
        for dec_i, dec_f in zip(decs[:-1], decs[1:]):

            mask = np.where(
                ((ra >= ra_i) & (ra < ra_f)
                & (dec >= dec_i) & (dec < dec_f))
            )
            patch_ids[mask] = patch_id

            patch_id += 1

    return patch_ids

def assignPatchesHEALPix(
        ra: np.ndarray,
        dec: np.ndarray,
        nside: int = 1024
    ) -> np.ndarray:
    """
    Assign patches using a HEALPix schema.
    For details see Górski et al. (2005).

    Parameters
    ----------
    ra : `~numpy.ndarray` (N)
        Right Ascension in degrees.
    dec : `~numpy.ndarray` (N)
        Declination in degrees.
    nside : int
        HEALPix nside parameter (must be a power of 2).

    Returns
    -------
    patch_ids : `~numpy.ndarray` (N)
        The patch ID for each coordinate.

    References
    ----------
    [1] Górski, K. M., Hivon, E., Banday, A. J., Wandelt, B. D., Hansen, F. K., Reinecke, M., & Bartelmann, M. (2005).
        HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere.
        The Astrophysical Journal, 622(2), 759. https://doi.org/10.1086/427976
    """
    patch_ids = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)

    return patch_ids