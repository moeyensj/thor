import os
import numpy as np
import pandas as pd
from typing import Optional
from astropy.time import Time

from .file_manager import FileManager

__all__ = [
    "_unpack_MPC_date",
    "__lookup_MPC",
    "convert_MPC_packed_dates",
    "pack_MPC_designation",
    "unpack_MPC_designation",
    "get_MPC_observatory_codes",
    "read_MPC_observatory_codes",
    "get_MPC_designation_files",
    "read_MPC_designation_files",
    "get_MPC_orbit_catalog",
    "read_MPC_orbit_catalog",
    "get_MPC_comet_catalog",
    "read_MPC_comet_catalog"
]

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_MAP = {BASE62[i] : i for i in range(len(BASE62))}

def _unpack_MPC_date(epoch_pf):
    # Taken from Lynne Jones' SSO TOOLS.
    # See https://minorplanetcenter.net/iau/info/PackedDates.html
    # for MPC documentation on packed dates.
    # Examples:
    #    1998 Jan. 18.73     = J981I73
    #    2001 Oct. 22.138303 = K01AM138303
    epoch_pf = str(epoch_pf)
    year = __lookup_MPC(epoch_pf[0])*100 + int(epoch_pf[1:3])
    month = __lookup_MPC(epoch_pf[3])
    day = __lookup_MPC(epoch_pf[4])
    isot_string = "{:d}-{:02d}-{:02d}".format(year, month, day)

    if len(epoch_pf) > 5:
        fractional_day = float("." + epoch_pf[5:])
        hours = int((24 * fractional_day))
        minutes = int(60 * ((24 * fractional_day) - hours))
        seconds = 3600 * (24 * fractional_day - hours - minutes / 60)
        isot_string += "T{:02d}:{:02d}:{:09.6f}".format(hours, minutes, seconds)

    return isot_string

def __lookup_MPC(x):
    # Convert the single character dates into integers.
    try:
        x = int(x)
    except ValueError:
        x = ord(x) - 55
    if x < 0 or x > 31:
        raise ValueError
    return x

def convert_MPC_packed_dates(pf_tt):
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
        isot_tt[i] = _unpack_MPC_date(epoch)

    epoch = Time(isot_tt, format="isot", scale="tt")
    return epoch.tt.mjd

def pack_MPC_designation(designation):
    """
    Pack a unpacked MPC designation. For example, provisional
    designation 1998 SS162 will be packed to J98SG2S. Permanent
    designation 323 will be packed to 00323.

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation : str
        MPC unpacked designation.

    Returns
    -------
    designation_pf : str
        MPC packed form designation.

    Raises
    ------
    ValueError : If designation cannot be packed.
    """
    is_numbered = True
    is_provisional = True
    is_survey = True
    is_packed = False

    # If the designation contains a dash it must be a
    # survey designation
    if "-" in designation:
        is_numbered = False
        is_provisional = False

    # Lets see if its a numbered object
    while is_numbered and not is_packed:
        try:
            number = int(designation)
            if number <= 99999:
                designation_pf = "{:05}".format(number)
            elif (number >= 100000) & (number <= 619999):
                ind = int(np.floor(number / 10000))
                designation_pf = "{}{:04}".format(BASE62[ind], number % 10000)
            else:
                x = number - 620000
                number_pf = []
                while x:
                    number_pf.append(BASE62[int(x % 62)])
                    x = int(x / 62)

                number_pf.reverse()
                designation_pf = "~{}".format("".join(number_pf).zfill(4))

            is_packed = True

        except:
            is_numbered = False

    # If its not numbered, maybe its a provisional designation
    while is_provisional and not is_packed:
        try:
            year = BASE62[int(designation[0:2])] + designation[2:4]
            letter1 = designation[5]
            letter2 = designation[6]
            cycle = designation[7:]

            cycle_pf = "00"
            if len(cycle) > 0:
                cycle = int(cycle)
                if cycle <= 99:
                    cycle_pf = str(cycle).zfill(2)
                else:
                    cycle_pf = BASE62[int(cycle / 10)] + str(cycle % 10)

            designation_pf = "{}{}{}{}".format(year, letter1, cycle_pf, letter2)
            is_packed = True

        except:
            is_provisional = False

    # If its a survey designation, deal with it
    while is_survey and not is_packed:
        try:
            number = designation[0:4]
            survey = designation[5:]

            if survey == "P-L":
                survey_pf = "PLS"

            if survey[0:2] == "T-":
                survey_pf = "T{}S".format(survey[2])

            designation_pf = "{}{}".format(survey_pf, number.zfill(4))
            is_packed = True

        except:
            is_survey = False

    # If at this point its not yet packed then something went wrong
    if not is_numbered and not is_provisional and not is_survey:
        err = (
            "Unpacked designation '{}' could not be packed.\n"
            "It could not be recognized as any of the following:\n"
            " - a numbered object (e.g. '3202', '203289', '3140113')\n"
            " - a provisional designation (e.g. '1998 SV127', '2008 AA360')\n"
            " - a survey designation (e.g. '2040 P-L', '3138 T-1')"
        )
        raise ValueError(err.format(designation))

    return designation_pf

def unpack_MPC_designation(designation_pf):
    """
    Unpack a packed MPC designation. For example, provisional
    designation J98SG2S will be unpacked to 1998 SS162. Permanent
    designation 00323 will be unpacked to 323.

    TODO: add support for comet and natural satellite designations

    Parameters
    ----------
    designation_pf : str
        MPC packed form designation.

    Returns
    -------
    designation : str
        MPC unpacked designation.

    Raises
    ------
    ValueError : If designation_pf cannot be unpacked.
    """
    is_numbered = True
    is_provisional = True
    is_survey = True
    is_unpacked = False

    while is_numbered and not is_unpacked:
        try:
            # Numbered objects (1 - 99999)
            if designation_pf.isdecimal():
                number = int(designation_pf)

            # Numbered objects (620000+)
            elif designation_pf[0] == "~":
                number = 620000
                number_pf = designation_pf[1:]
                for i, c in enumerate(number_pf):
                    power = (len(number_pf) - (i + 1))
                    number += BASE62_MAP[c] * (62**power)

            # Numbered objects (100000 - 619999)
            else:
                number = BASE62_MAP[designation_pf[0]] * 10000 + int(designation_pf[1:])

            designation = str(number)
            is_unpacked = True

        except:
            is_numbered = False

    while is_provisional and not is_unpacked:
        try:
            year = str(BASE62_MAP[designation_pf[0]] * 100 + int(designation_pf[1:3]))
            letter1 = designation_pf[3]
            letter2 = designation_pf[6]
            cycle1 = designation_pf[4]
            cycle2 = designation_pf[5]

            number = int(BASE62_MAP[cycle1]) * 10 + BASE62_MAP[cycle2]
            if number == 0:
                number = ""

            designation = "{} {}{}{}".format(year, letter1, letter2, number)
            is_unpacked = True

        except:
            is_provisional = False

    while is_survey and not is_unpacked:
        try:
            number = int(designation_pf[3:8])
            survey_pf = designation_pf[0:3]

            if survey_pf == "PLS":
                survey = "P-L"

            if survey_pf[0] == "T" and survey_pf[2] == "S":
                survey = "T-{}".format(survey_pf[1])

            designation = "{} {}".format(number, survey)
            is_unpacked = True

        except:
            is_survey = False

    if not is_numbered and not is_provisional and not is_survey:
        err = (
            "Packed form designation '{}' could not be unpacked.\n"
            "It could not be recognized as any of the following:\n"
            " - a numbered object (e.g. '03202', 'K3289', '~AZaz')\n"
            " - a provisional designation (e.g. 'J98SC7V', 'K08Aa0A')\n"
            " - a survey designation (e.g. 'PLS2040', 'T1S3138')"
        )
        raise ValueError(err.format(designation_pf))

    return designation

def get_MPC_observatory_codes():
    """
    Downloads the JSON-formatted MPC observatory codes file. Checks if a newer version of the file exists online, if so,
    replaces the previously downloaded file if available.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    file_manager = FileManager()
    url = "https://minorplanetcenter.net/Extended_Files/obscodes_extended.json.gz"
    file_manager.download(url, sub_directory="mpc")
    return

def read_MPC_observatory_codes(observatory_codes: Optional[str] = None):
    """
    Reads the JSON-formatted MPC observatory codes file.

    Parameters
    ----------
    observatory_codes : str, optional
        Path to file

    Returns
    -------
    observatories : `~numpy.ndarray` (N)
        Structured array containing the observatory codes and the
        geodetic terms for each observatory.

    See Also
    --------
    `~thor.utils.mpc.get_MPC_observatory_codes` : Downloads the MPC observatory codes file.
    """
    if observatory_codes is None:
        file_manager = FileManager()
        observatory_codes = file_manager.log["obscodes_extended.json.gz"]["location"]

    observatories_df = pd.read_json(observatory_codes, orient="index", precise_float=True)
    observatories_df.rename(columns={
        "Longitude" : "longitude_deg",
        "Name" : "name"},
        inplace=True
    )
    observatories_df.index.name = 'code'
    observatories_df.reset_index(inplace=True)

    observatories = np.zeros(
        len(observatories_df),
        dtype={
            "names" : ("code", "longitude_deg", "cos", "sin", "name"),
            "formats" : ("U3", np.float64, np.float64, np.float64, "U60")
        },
    )
    observatories["code"] = observatories_df["code"].values
    observatories["longitude_deg"] = observatories_df["longitude_deg"].values
    observatories["cos"] = observatories_df["cos"].values
    observatories["sin"] = observatories_df["sin"].values
    observatories["name"] = observatories_df["name"].values

    return observatories

def get_MPC_designation_files():
    """
    Downloads the JSON-formatted MPC designation files (both the unpacked and packed versions).
    Checks if a newer version of the files exist online, if so,
    replaces the previously downloaded files if available.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    file_manager = FileManager()
    url = "https://www.minorplanetcenter.net/Extended_Files/mpc_ids.json.gz"
    file_manager.download(url, sub_directory="mpc")
    url = "https://www.minorplanetcenter.net/Extended_Files/mpc_ids_packed.json.gz"
    file_manager.download(url, sub_directory="mpc")
    return

def read_MPC_designation_files(mpcDesignationsFile=None, mpcPackedDesignationsFile=None):
    """
    Reads the JSON-formatted MPC designation files (both the unpacked and packed forms).

    Parameters
    ----------
    mpcDesignationsFile : str, optional
        Path to file
    mpcPackedDesignationsFile : str, optional
        Path to file

    Returns
    -------
    designations : `~pandas.DataFrame`
        DataFrame of MPC designations
    designations_pf : `~pandas.DataFrame`
        DataFrame of MPC packed form designations

    See Also
    --------
    `~thor.utils.mpc.get_MPC_designation_files` : Downloads the JSON-formatted MPC designation files.
    """
    if mpcDesignationsFile is None:
        file_manager = FileManager()
        mpcDesignationsFile = file_manager.log["mpc_ids.json.gz"]["location"]

    if mpcPackedDesignationsFile is None:
        file_manager = FileManager()
        mpcPackedDesignationsFile = file_manager.log["mpc_ids_packed.json.gz"]["location"]

    designations = pd.read_json(mpcDesignationsFile, orient='index')
    designations = pd.DataFrame(designations.stack(), columns=["other_designations"])
    designations.reset_index(level=1, inplace=True, drop=True)
    designations.index.name = "designation"
    designations.reset_index(inplace=True)

    designations_pf = pd.read_json(mpcPackedDesignationsFile, orient='index')
    designations_pf = pd.DataFrame(designations_pf.stack(), columns=["other_designations_pf"])
    designations_pf.reset_index(level=1, inplace=True, drop=True)
    designations_pf.index.name = "designation_pf"
    designations_pf.reset_index(inplace=True)

    return designations, designations_pf

def get_MPC_orbit_catalog():
    """
    Downloads the JSON-formatted extended MPC orbit catalog. Checks if a newer version of the file exists online, if so,
    replaces the previously downloaded file if available.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    file_manager = FileManager()
    url = "https://www.minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"
    file_manager.download(url, "mpc")
    return

def read_MPC_orbit_catalog(mpc_orbit_catalog=None):
    """
    Reads the JSON-formatted extended MPC orbit catalog.

    Parameters
    ----------
    mpc_orbit_catalog : str, optional
        Path to file

    Returns
    -------
    mpcorb : `~pandas.DataFrame`
        DataFrame of MPC minor planet orbits.

    See Also
    --------
    `~thor.utils.mpc.get_MPC_orbit_catalog` : Downloads the extended MPC orbit catalog.
    """
    if mpc_orbit_catalog is None:
        file_manager = FileManager()
        mpc_orbit_catalog = file_manager.log["mpcorb_extended.json.gz"]["location"]

    mpcorb = pd.read_json(mpc_orbit_catalog, precise_float=True)

    # Rename columns, include units where possible
    mpcorb.rename(columns={
        "Number" : "number",
        "Name" :  "name",
        "Principal_desig" : "provisional_designation",
        "Other_desigs" : "other_provisional_designations",
        "Epoch" : "jd_tt",
        "a" : "a_au",
        "e" : "e",
        "i" : "i_deg",
        "Node" : "ascNode_deg",
        "Peri" : "argPeri_deg",
        "M" : "meanAnom_deg",
        "n" : "mean_motion_deg_p_day",
        "H" : "H_mag",
        "G" : "G",
        "Tp" : "tPeri_jd_tt",
        "Orbital_period" : "period_yr",
        "Perihelion_dist" : "perihelion_dist_au",
        "Aphelion_dist" : "aphelion_dist_au",
        "Semilatus_rectum" : "p_au",
        "Synodic_period" : "synodic_period_yr",
        "Orbit_type" : "orbit_type",
        "Num_obs" : "num_obs",
        "Last_obs" : "last_obs",
        "rms" : "rms_arcsec",
        "U" : "uncertainty_param",
        "Arc_years" : "arc_yr",
        "Arc_length" : "arc_days",
        "Num_opps" : "num_oppos",
        "Perturbers" : "perturbers1",
        "Perturbers_2" : "perturbers2",
        "Hex_flags" : "hex_flags",
        "NEO_flag" : "neo_flag",
        "One_km_NEO_flag" : "1km_neo_flag",
        "PHA_flag" : "pha_flag",
        "Critical_list_numbered_object_flag" : "critical_list_flag",
        "One_opposition_object_flag" : "1_oppo_flag",
        "Ref" : "reference",
        "Computer" : "computer"
    }, inplace=True)

    # Add MJDs from JDs
    mpcorb["mjd_tt"] = Time(mpcorb["jd_tt"].values, format="jd", scale="tt").tt.mjd
    mpcorb["tPeri_mjd_tt"] = Time(mpcorb["tPeri_jd_tt"].values, format="jd", scale="tt").tt.mjd

    # Drop redundant columns
    mpcorb.drop(columns=["jd_tt", "tPeri_jd_tt"], inplace=True)

    # Create a designation column, if an asteroid is numbered use that as a designation if not use the provisional designation
    mpcorb.loc[~mpcorb["number"].isna(), "designation"] = mpcorb[~mpcorb["number"].isna()]["number"].str.replace('[()]', '', regex=True).values
    mpcorb.loc[mpcorb["designation"].isna(), "designation"] = mpcorb[mpcorb["designation"].isna()]["provisional_designation"].values

    # Arrange columns
    columns = [
        "designation",
        "number",
        "name",
        "provisional_designation",
        "other_provisional_designations",
        "mjd_tt",
        "a_au",
        "e",
        "i_deg",
        "ascNode_deg",
        "argPeri_deg",
        "meanAnom_deg",
        "mean_motion_deg_p_day",
        "H_mag",
        "G",
        "uncertainty_param",
        "tPeri_mjd_tt",
        "p_au",
        "period_yr",
        "perihelion_dist_au",
        "aphelion_dist_au",
        "synodic_period_yr",
        "num_obs",
        "last_obs",
        "rms_arcsec",
        "num_oppos",
        "arc_yr",
        "arc_days",
        "orbit_type",
        "neo_flag",
        "1km_neo_flag",
        "pha_flag",
        "1_oppo_flag",
        "critical_list_flag",
        "hex_flags",
        "perturbers1",
        "perturbers2",
        "reference",
        "computer",
    ]
    mpcorb = mpcorb[columns]

    return mpcorb

def get_MPC_comet_catalog():
    """
    Downloads the JSON-formatted MPC comet orbit catalog. Checks if a newer version of the file exists online, if so,
    replaces the previously downloaded file if available.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    file_manager = FileManager()
    url = "https://www.minorplanetcenter.net/Extended_Files/cometels.json.gz"
    file_manager.download(url, sub_directory="mpc")
    return

def read_MPC_comet_catalog(mpc_comet_catalog: Optional[str] = None):
    """
    Reads the JSON-formatted MPC comet catalog.

    Parameters
    ----------
    mpc_comet_catalog : str, optional
        Path to file

    Returns
    -------
    mpcorb_comets : `~pandas.DataFrame`
        DataFrame of MPC comet orbits.

    See Also
    --------
    `~thor.utils.mpc.get_MPC_comet_catalog` : Downloads the MPC comet catalog.
    """
    file_manager = FileManager()
    if mpc_comet_catalog is None:
        mpc_comet_catalog = file_manager.log["cometels.json.gz"]["location"]
    mpcorb_comets = pd.read_json(mpc_comet_catalog, precise_float=True)

    mpcorb_comets.rename(columns={
         "Orbit_type" : "orbit_type",
         "Provisional_packed_desig" : "provisional_designation_pf",
         "Year_of_perihelion" : "tPeri_yr",
         "Month_of_perihelion" : "tPeri_month",
         "Day_of_perihelion" : "tPeri_day",
         "Perihelion_dist" : "q_au",
         "e": "e",
         "Peri" : "argPeri_deg",
         "Node" : "ascNode_deg",
         "i" : "i_deg",
         "Epoch_year" : "epoch_yr",
         "Epoch_month" : "epoch_month",
         "Epoch_day" : "epoch_day",
         "H" : "H_mag",
         "G" : "G",
         "Designation_and_name" : "designation_name",
         "Ref" : "reference",
         "Comet_num" : "comet_number"
        },
        inplace=True
    )

    # Update time of perihelion passage to be an MJD
    yr = mpcorb_comets["tPeri_yr"].values
    month = mpcorb_comets["tPeri_month"].values
    day = mpcorb_comets["tPeri_day"].values
    yr_month = ["{}-{:02d}-01T00:00:00".format(y, m) for y, m in zip(yr, month)]
    t_peri = Time(yr_month, format="isot", scale="tt")
    t_peri += day
    mpcorb_comets["tPeri_mjd_tt"] = t_peri.tt.mjd

    # Update orbit epoch to be an MJD
    mask = (~mpcorb_comets["epoch_yr"].isna())
    yr = mpcorb_comets[mask]["epoch_yr"].values.astype(int)
    month = mpcorb_comets[mask]["epoch_month"].values.astype(int)
    day = mpcorb_comets[mask]["epoch_day"].values
    yr_month = ["{:d}-{:02d}-01T00:00:00".format(y, m) for y, m in zip(yr, month)]
    epoch = Time(yr_month, format="isot", scale="tt")
    epoch += day
    mpcorb_comets.loc[mask, "mjd_tt"] = epoch.tt.mjd

    # Remove redundant columns
    mpcorb_comets.drop(columns=[
        "epoch_yr",
        "epoch_month",
        "epoch_day",
        "tPeri_yr",
        "tPeri_month",
        "tPeri_day"
        ],
        inplace=True
    )

    # Convert comet number to strings
    mpcorb_comets.loc[~mpcorb_comets["comet_number"].isna(), "comet_number"] = mpcorb_comets[~mpcorb_comets["comet_number"].isna()]["comet_number"].apply(lambda x: str(int(x))).values

    # Split designation_name into designation
    mpcorb_comets["designation"] = mpcorb_comets["designation_name"].str.split(" [()]", expand=True)[0].values

    # Remove name from numbered comets
    mpcorb_comets.loc[~mpcorb_comets["comet_number"].isna(), "designation"] = mpcorb_comets[~mpcorb_comets["comet_number"].isna()]["designation_name"].str.split("/", expand=True)[0].values

    # Arrange columns
    columns = [
        "designation",
        "designation_name",
        "comet_number",
        "provisional_designation_pf",
        "mjd_tt",
        "q_au",
        "e",
        "i_deg",
        "ascNode_deg",
        "argPeri_deg",
        "tPeri_mjd_tt",
        "H_mag",
        "G",
        "orbit_type",
        "reference"
    ]
    mpcorb_comets = mpcorb_comets[columns]

    return mpcorb_comets