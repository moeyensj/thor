import numpy as np

from ..patches import assignPatchesSquare

def test_assignPatchesSquare():

    ra_width = 15
    dec_width = 15
    num_ra = 360 / ra_width
    num_dec = 180 / dec_width

    # Create fake observations at the midpoint of each bin
    ra =  np.arange(0., 360, ra_width)
    ra += ra_width / 2
    dec = np.arange(-90, 90., dec_width)
    dec += dec_width / 2

    dec, ra = np.meshgrid(dec, ra)
    dec = dec.flatten()
    ra = ra.flatten()
    patch_ids_desired = np.arange(0, len(dec), dtype=int)

    patch_ids = assignPatchesSquare(ra, dec, ra_width=ra_width, dec_width=dec_width)

    # Check that the patch IDs match the expected order
    # (increasing Dec first, then increasing RA)
    np.testing.assert_equal(patch_ids_desired, patch_ids)
