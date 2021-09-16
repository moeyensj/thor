import warnings
import numpy as np
import pandas as pd
from astropy.time import Time

__all__ = [
    "UNKNOWN_ID_REGEX",
    "preprocessObservations",
]

UNKNOWN_ID_REGEX = "^u[0-9]{12}$"

def preprocessObservations(
        observations,
        column_mapping,
        astrometric_errors=None,
        mjd_scale="utc"
    ):
    """
    Create two seperate data frames: one with all observation data needed to run THOR stripped of
    object IDs and the other with known object IDs and attempts to attribute unknown observations to
    the latest catalog of known objects from the MPC.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing at minimum a column of observation IDs, exposure times in MJD (with scale
        set by mjd_scale), RA in degrees, Dec in degrees, 1-sigma error in RA in degrees, 1-sigma error in
        Dec in degrees and the observatory code.
    column_mapping : dict
        Dictionary containing internal column names as keys mapped to column names in the data frame as values.
        Should include the following:
        {# Internal : # External
            "obs_id" : column name or None,
            "mjd" : column name,
            "RA_deg" : column name,
            "Dec_deg" : column name,
            "RA_sigma_deg" : column name or None,
            "Dec_sigma_deg" : column name or None,
            "observatory_code" : column name,
            "obj_id" : column name or None,
            "mag" : optional, column name or None,
            "mag_sigma" : optional, column name or None,
            "filter" : optional, column name or None,
            "astrometric_catalog" : optional, column name or None,
        }
        Description of columns and their assumed values:
            'obs_id' : column name or None
                Observation IDs as type string. If None, THOR will assign
                an observation ID to each observation.
            'mjd' : column name
                Observation time in MJD, the input time scale can be set with the
                'time_scale' parameter. Time scale will be converted if not in UTC.
            'RA_deg' : column name
                Topocentric J2000 Right Ascension in degrees.
            'Dec_deg' : column name
                Topocentric J2000 Declination in degrees.
            'RA_sigma_deg' : column name or None
                 1-sigma astrometric uncertainty in RA in degrees.
                 If certain or all observations are missing astrometric errors, use
                 the 'astrometric_errors' parameter to configure defaults for all observatories
                 or for each observatory individually. If None, THOR will use the 'astrometric_error'
                 parameter to assign errors.
            'Dec_sigma_deg' : column name or None
                 1-sigma astrometric uncertainty in Dec in degrees.
                 If certain or all observations are missing astrometric errors, use
                 the 'astrometric_errors' parameter to configure defaults for all observatories
                 or for each observatory individually. If None, THOR will use the 'astrometric_error'
                 parameter to assign errors.
            'observatory_code' : column name
                The MPC observatory code from which each observation was made. THOR currently
                only supports ground-based observatories.
            'obj_id' : column name or None
                If known, the designation in unpacked or packed form. If unknown, object ID should be
                set to 'NaN'. If None, THOR will assume no observations have been associated.
            'mag' : optional, column name or None
                Observed magnitude. Magnitudes are currently unused by THOR but may be convenient to have
                for visual inspection of results.
            'mag_sigma' : optional, column name or None.
                1-sigma photometric uncertainty in magnitudes.
            'filter' : optional, column name or None.
                The bandpass or filter with which the observation was made.
            'astrometric_catalog' : optional, column name or None.
                Astrometric catalog with which astrometric measurements were calibrated. Unused by THOR outside of
                creating ADES files from recoveries and discoveries.
            'night_id' : optional, column_name or None.
                ID representing the night on which an observation was made. Useful for filter for observations on
                single nights rather than using the observation time.
    mjd_scale : str, optional
        Time scale of the input MJD exposure times ("utc", "tdb", etc...)

    Returns
    -------
    preprocessed_observations : `~pandas.DataFrame`
        DataFrame with observations in the format required by THOR.
    preprocessed_attributions : `~pandas.DataFrame`
        DataFrame containing associations, any observations with no known label
        will be assigned a unique unknown ID with regex pattern "^u[0-9]{12}$".

    Raises
    ------
    ValueError
        If the astrometric_errors parameter is not of type list or dictionary,
        or if the errors are not correctly defined.

    Warns
    -----
    UserWarning:
        If the observation ID, object_ID, or astrometric error columns are not
        present in the column_mapping dictionary.
    """
    # Required columns THOR needs
    cols = [
        "obs_id",
        "mjd",
        "RA_deg",
        "Dec_deg",
        "RA_sigma_deg",
        "Dec_sigma_deg",
        "observatory_code",
        "obj_id"
    ]
    # Optional columns that can be used for filtering
    # and ADES file production
    optional_cols = [
        # ADES Columns
        "mag",
        "mag_sigma",
        "filter",
        "astrometric_catalog",

        # Useful non-ADES columns
        "night_id"
    ]

    # Check if observation IDs need to be assigned
    assign_obs_ids = False
    if column_mapping["obs_id"] == None:
        warning = (
            "No observation ID column defined in the column_mapping dictionary.\n"
            "Assigning observation IDs...\n"
        )
        warnings.warn(
            warning,
            UserWarning
        )
        assign_obs_ids = True
        cols.remove("obs_id")

    # Check if object IDs need to be assigned
    assign_obj_ids = False
    if column_mapping["obj_id"] == None:
        warning = (
            "No object ID column defined in the column_mapping dictionary.\n"
            "Assuming no observations have been associated with a known object...\n"
        )
        warnings.warn(
            warning,
            UserWarning
        )
        assign_obj_ids = True
        cols.remove("obj_id")

    # Check if astrometric errors need to be added
    use_astrometric_errors = False
    if (column_mapping["RA_sigma_deg"] == None) and (column_mapping["Dec_sigma_deg"] == None):
        warning = (
            "No astrometric error columns defined in the column_mapping dictionary.\n"
            "Using 'astrometric_errors' parameter to assign errors...\n"
        )
        warnings.warn(
            warning,
            UserWarning
        )
        use_astrometric_errors = True
        cols.remove("RA_sigma_deg")
        cols.remove("Dec_sigma_deg")


    # Create a copy of the relevant columns in observations
    # Add any optional columns that may have been provided by the user
    obs_cols = [column_mapping[c] for c in cols]
    added_cols = []
    for c in optional_cols:
        if c in column_mapping.keys():
            obs_cols.append(column_mapping[c])
            added_cols.append(c)
    preprocessed_observations = observations[obs_cols].copy()

    # Rename preprocessed observation columns to those expected by THOR
    # (involves inverting the column_mapping dictionary and removing any potential
    # None values passed by the user)
    column_mapping_inv = {v : k for k, v in column_mapping.items()}
    if None in column_mapping_inv.keys():
        column_mapping_inv.pop(None)
    preprocessed_observations.rename(
        columns=column_mapping_inv,
        inplace=True
    )

    if use_astrometric_errors:
        if type(astrometric_errors) == list:
            if len(astrometric_errors) != 2:
                err = (
                    "astrometric_errors list is not of length 2."
                )
            else:
                preprocessed_observations.loc[:, "RA_sigma_deg"] = astrometric_errors[0]
                preprocessed_observations.loc[:, "Dec_sigma_deg"] = astrometric_errors[1]

        elif type(astrometric_errors) == dict:
            for code, errors in astrometric_errors.items():
                if len(errors) != 2:
                    err = (
                        "Astrometric errors for observatory {} should be a list of length 2 with\n"
                        "the 1-sigma astrometric uncertainty in RA as the first element and the\n"
                        "1-sigma astrometric uncertainty in Dec as the second element."
                    )
                    raise ValueError(err.format(code))
                else:
                    observatory_mask = preprocessed_observations["observatory_code"].isin([code])
                    preprocessed_observations.loc[observatory_mask, "RA_sigma_deg"] = errors[0]
                    preprocessed_observations.loc[observatory_mask, "Dec_sigma_deg"] = errors[1]

        else:
            err = (
                "'astrometric_errors' should be one of {None, list, dict}.\n"
                "If None, then the given observations must have the ra_sigma_deg\n"
                "  and dec_sigma_deg columns.\n"
                "If a dictionary, then each observatory code present observations in\n"
                "  the observations must have a corresponding key with a list of length 2\n"
                "  as their values. The first element in the list is assumed to be the 1-sigma\n"
                "  astrometric error in RA, while the second is assumed to be the same but in Dec.\n"
                "If a list, then the first element in the list is assumed to be the 1-sigma\n"
                "  astrometric error in RA, while the second is assumed to be the same but in Dec.\n"
                "  Each observation will be given these errors regardless of if one is present or not.\n"
            )
            raise ValueError(err)

    # Make sure all observations have astrometric errors
    missing_codes = preprocessed_observations[(
        (preprocessed_observations["RA_sigma_deg"].isna())
        | (preprocessed_observations["Dec_sigma_deg"].isna())
    )]["observatory_code"].unique()

    if len(missing_codes) > 0:
        err = (
            "Missing astrometric errors for observations from:\n"
            "  {}\n"
        )
        raise ValueError(err.format(", ".join(missing_codes)))

    # Make sure all observations are given in UTC, if not convert to UTC
    if mjd_scale != "utc":
        mjds = Time(
            preprocessed_observations["mjd"].values,
            format="mjd",
            scale=mjd_scale
        )
        preprocessed_observations["mjd"] = mjds.utc.mjd

    # Add _utc to mjd column name
    preprocessed_observations.rename(
        columns={
            "mjd" : "mjd_utc"
        },
        inplace=True
    )

    # Make sure that the observations are sorted by observation time
    preprocessed_observations.sort_values(
        by=["mjd_utc"],
        inplace=True,
        ignore_index=True
    )
    # Assign obervation IDs if needed
    if assign_obs_ids:
        preprocessed_observations.loc[:, "obs_id"] = ["obs{:09d}".format(i) for i in range(len(preprocessed_observations))]
    else:
        if type(preprocessed_observations["obs_id"]) != object:
            warn = ("Observation IDs should be of type string, converting...")
            warnings.warn(warn)
            preprocessed_observations["obs_id"] = preprocessed_observations["obs_id"].astype(str)

    # Assign object IDs if needed
    if assign_obj_ids:
        # This must match UNKNOWN_ID_REGEX
        preprocessed_observations.loc[:, "obj_id"] = [f"u{i:012d}" for i in range(len(preprocessed_observations))]
    else:
        if type(preprocessed_observations["obj_id"]) != object:
            warn = ("Object IDs should be of type string, converting...")
            warnings.warn(warn)
            num_unassociated = len(preprocessed_observations[preprocessed_observations["obj_id"].isna()])
            # This must match UNKNOWN_ID_REGEX
            preprocessed_observations.loc[preprocessed_observations["obj_id"].isna(), "obj_id"] = [f"u{i:012d}" for i in range(num_unassociated)]
            preprocessed_observations["obj_id"] = preprocessed_observations["obj_id"].astype(str)

    # Split observations into two dataframes (make THOR run only on completely blind observations)
    preprocessed_associations = preprocessed_observations[[
        "obs_id",
        "obj_id"
    ]].copy()
    cols_sorted = [
        "obs_id",
        "mjd_utc",
        "RA_deg",
        "Dec_deg",
        "RA_sigma_deg",
        "Dec_sigma_deg",
        "observatory_code"
    ]
    cols_sorted += added_cols
    preprocessed_observations = preprocessed_observations[cols_sorted]

    return preprocessed_observations, preprocessed_associations