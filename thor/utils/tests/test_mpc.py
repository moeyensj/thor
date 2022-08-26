import numpy as np
from astropy.time import Time

from ..mpc import convert_MPC_packed_dates
from ..mpc import pack_MPC_designation
from ..mpc import unpack_MPC_designation

### Tests last updated: 2022-08-26

DESIGNATIONS_UP2P = {
    # Packed : Unpacked
    # Taken from: https://www.minorplanetcenter.net/iau/info/PackedDes.html

    # Provisional Minor Planet Designations
    "1995 XA" : "J95X00A",
    "1995 XL1"  : "J95X01L",
    "1995 FB13" : "J95F13B",
    "1998 SQ108" : "J98SA8Q",
    "1998 SV127" :  "J98SC7V",
    "1998 SS162" : "J98SG2S",
    "2099 AZ193" : "K99AJ3Z",
    "2008 AA360" : "K08Aa0A",
    "2007 TA418" : "K07Tf8A",

    # Provisional Minor Planet Designations (Surveys)
    "2040 P-L" : "PLS2040",
    "3138 T-1" : "T1S3138",
    "1010 T-2" : "T2S1010",
    "4101 T-3" : "T3S4101",

    # Permanent Minor Planet Designations
    "3202" : "03202",
    "50000" : "50000",
    "100345" : "A0345",
    "360017" : "a0017",
    "203289" : "K3289",
    "620000" : "~0000",
    "620061" : "~000z",
    "3140113" : "~AZaz",
    "15396335" : "~zzzz"
}
DESIGNATIONS_P2UP = {v : k for k, v in DESIGNATIONS_UP2P.items()}


def test_convert_MPC_packed_dates():
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

    mjd_tt = convert_MPC_packed_dates(pf_tt)
    mjd = Time(isot_tt, format="isot", scale="tt")

    np.testing.assert_equal(mjd_tt, mjd.tt.mjd)
    return

def test_unpack_MPC_designation():
    # Test unpacking of packed form designations
    for designation_pf, designation in DESIGNATIONS_P2UP.items():
        assert unpack_MPC_designation(designation_pf) == designation

def test_pack_MPC_designation():
    # Test packing of unpacked designations
    for designation, designation_pf in DESIGNATIONS_UP2P.items():
        assert pack_MPC_designation(designation) == designation_pf