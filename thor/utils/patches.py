import numpy as np

__all__ = [
    "assignPatchesSquare"
]

def assignPatchesSquare(ra, dec, ra_width=15, dec_width=15):
    """
    Assign a patch ID to each observation where a patch is a square region
    on the sky plane of ra_width in RA and of dec_width in Dec.

    RA must be between 0 and 360 degrees.
    Dec must be between -90 and 90 degrees.

    Patches are created first in increasing Declination and second in increasing
    Right Ascension. The first patch will border RA of 0 and Dec of -90, while
    the last patch will border RA of 360 and Dec of 90.

    Parameters
    ----------
    ra : `~numpy.ndarrray` (N)
        Right Ascension in degrees.
    dec : `~numpy.ndarrray` (N)
        Declination in degrees.
    ra_width : float
        Width of patch in RA in degrees.
    dec_with : float
        Width of patch in Dec in degrees.

    Returns
    -------
    patch_ids : `~numpy.ndarray` (N)
        The patch ID for each observation.
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