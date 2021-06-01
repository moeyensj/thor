import warnings
import numpy as np
import pandas as pd
from astropy.time import Time

__all__ = [
    "preprocessObservations",
    "findAverageOrbits",
]

def preprocessObservations(
        observations,
        column_mapping,
        astrometric_errors=None,
        mjd_scale="utc",
        attribution=False
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
            'observatory_code' : column_name
                The MPC observatory code from which each observation was made. THOR currently
                only supports ground-based observatories.
            'obj_id' : column_name or None
                If known, the designation in unpacked or packed form. If unknown, object ID should be
                set to 'NaN'. If None, THOR will assume no observations have been associated.
    mjd_scale : str, optional
        Time scale of the input MJD exposure times ("utc", "tdb", etc...)
    attribution : bool, optional
        Place holder boolean to trigger attribution

    Returns
    -------
    preprocessed_observations : `~pandas.DataFrame`
        DataFrame with observations in the format required by THOR.
    preprocessed_attributions : `~pandas.DataFrame`
        DataFrame containing truths.

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
    obs_cols = [column_mapping[c] for c in cols]
    preprocessed_observations = observations[obs_cols].copy()

    # Rename preprocessed observation columns to those expected by THOR
    # (involves inverting the column_mapping dictionary and removing any potential
    # None values passed by the user)
    column_mapping_inv = {v : k for k, v in column_mapping.items()}
    if None in column_mapping_inv.keys():
        column_mapping_inv.pop(None)
    preprocessed_observations.rename(
        columns=column_mapping_inv,
        inplace=True)

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
        inplace=True
    )

    # Reset index after sort
    preprocessed_observations.reset_index(
        inplace=True,
        drop=True
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
        preprocessed_observations.loc[:, "obj_id"] = "None"
    else:
        if type(preprocessed_observations["obj_id"]) != object:
            warn = ("Object IDs should be of type string, converting...")
            warnings.warn(warn)
            preprocessed_observations.loc[preprocessed_observations["obj_id"].isna(), "obj_id"] = "None"
            preprocessed_observations["obj_id"] = preprocessed_observations["obj_id"].astype(str)

    # Split observations into two dataframes (make THOR run only on completely blind observations)
    preprocessed_associations = preprocessed_observations[[
        "obs_id",
        "obj_id"
    ]].copy()
    preprocessed_observations = preprocessed_observations[[
        "obs_id",
        "mjd_utc",
        "RA_deg",
        "Dec_deg",
        "RA_sigma_deg",
        "Dec_sigma_deg",
        "observatory_code",
    ]]

    return preprocessed_observations, preprocessed_associations

COLUMN_MAPPING = {
        ### Observation Parameters

        # Observation ID
        "obs_id" : "obsId",

        # Exposure time
        "exp_mjd" : "exp_mjd",

        # Visit ID
        "visit_id" : "visitId",

        # Field ID
        "field_id" : "fieldId",

        # Field RA in degrees
        "field_RA_deg" : "fieldRA_deg",

        # Field Dec in degrees
        "field_Dec_deg" : "fieldDec_deg",

        # Night number
        "night": "night",

        # RA in degrees
        "RA_deg" : "RA_deg",

        # Dec in degrees
        "Dec_deg" : "Dec_deg",

        # Observatory code
        "observatory_code" : "code",

        # Observer's x coordinate in AU
        "obs_x_au" : "HEclObsy_X_au",

        # Observer's y coordinate in AU
        "obs_y_au" : "HEclObsy_Y_au",

        # Observer's z coordinate in AU
        "obs_z_au" : "HEclObsy_Z_au",

        # Magnitude (UNUSED)
        "mag" : "VMag",

        ### Truth Parameters

        # Object name
        "name" : "designation",

        # Observer-object distance in AU
        "Delta_au" : "Delta_au",

        # Sun-object distance in AU (heliocentric distance)
        "r_au" : "r_au",

        # Object's x coordinate in AU
        "obj_x_au" : "HEclObj_X_au",

        # Object's y coordinate in AU
        "obj_y_au" : "HEclObj_Y_au",

        # Object's z coordinate in AU
        "obj_z_au" : "HEclObj_Z_au",

        # Object's x velocity in AU per day
        "obj_dx/dt_au_p_day" : "HEclObj_dX/dt_au_p_day",

        # Object's y velocity in AU per day
        "obj_dy/dt_au_p_day" : "HEclObj_dY/dt_au_p_day",

        # Object's z velocity in AU per day
        "obj_dz/dt_au_p_day" : "HEclObj_dZ/dt_au_p_day",

        # Semi-major axis
        "a_au" : "a_au",

        # Inclination
        "i_deg" : "i_deg",

        # Eccentricity
        "e" : "e",
    }


def findAverageOrbits(
        observations,
        orbits,
        d_values=None,
        element_type="keplerian",
        column_mapping=COLUMN_MAPPING
    ):
    """
    Find the object with observations that represents
    the most average in terms of cartesian velocity and the
    heliocentric distance. Assumes that a subset of the designations in the orbits
    dataframe are identical to at least some of the designations in the observations
    dataframe. No propagation is done, so the orbits need to be defined at an epoch near
    the time of observations, for example like the midpoint or start of a two-week window.

    Parameters
    ----------
    observations : `~pandas.DataFrame`
        DataFrame containing observations.
    orbits : `~pandas.DataFrame`
        DataFrame containing orbits for each unique object in observations.
    d_values : {list (N>=2), None}, optional
        If None, will find average orbit in all of observations. If a list, will find an
        average orbit between each value in the list. For example, passing dValues = [1.0, 2.0, 4.0] will
        mean an average orbit will be found in the following bins: (1.0 <= d < 2.0), (2.0 <= d < 4.0).
    element_type : {'keplerian', 'cartesian'}, optional
        Find average orbits using which elements. If 'keplerian' will use a-e-i for average,
        if 'cartesian' will use r, v.
        [Default = 'keplerian']
    verbose : bool, optional
        Print progress statements?
        [Default = True]
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names.
        [Default = `~thor.Config.COLUMN_MAPPING`]

    Returns
    -------
    orbits : `~pandas.DataFrame`
        DataFrame with name, r, v, exposure time, and sky-plane location of the average orbit in each bin of r.
    """
    if element_type == "keplerian":
        d_col = column_mapping["a_au"]
    elif element_type == "cartesian":
        d_col = column_mapping["r_au"]
    else:
        err = (
            "element_type should be one of {'keplerian', 'cartesian'}"
        )
        raise ValueError(err)

    dataframe = pd.merge(orbits, observations, on=column_mapping["name"]).copy()
    dataframe.reset_index(inplace=True, drop=True)

    d_bins = []
    if d_values != None:
        for d_i, d_f in zip(d_values[:-1], d_values[1:]):
            d_bins.append(dataframe[(dataframe[d_col] >= d_i) & (dataframe[d_col] < d_f)])
    else:
        d_bins.append(dataframe)

    average_orbits = []

    for i, obs in enumerate(d_bins):
        if len(obs) == 0:
            # No real objects

            orbit = pd.DataFrame({"orbit_id" : i + 1,
                column_mapping["exp_mjd"] : np.NaN,
                column_mapping["obj_x_au"] : np.NaN,
                column_mapping["obj_y_au"] : np.NaN,
                column_mapping["obj_z_au"] : np.NaN,
                column_mapping["obj_dx/dt_au_p_day"] : np.NaN,
                column_mapping["obj_dy/dt_au_p_day"] : np.NaN,
                column_mapping["obj_dz/dt_au_p_day"] : np.NaN,
                column_mapping["RA_deg"] : np.NaN,
                column_mapping["Dec_deg"] : np.NaN,
                column_mapping["r_au"] : np.NaN,
                column_mapping["a_au"] : np.NaN,
                column_mapping["i_deg"] : np.NaN,
                column_mapping["e"] : np.NaN,
                column_mapping["name"]: np.NaN}, index=[0])
            average_orbits.append(orbit)
            continue

        if element_type == "cartesian":

            rv = obs[[
                column_mapping["obj_dx/dt_au_p_day"],
                column_mapping["obj_dy/dt_au_p_day"],
                column_mapping["obj_dz/dt_au_p_day"],
                column_mapping["r_au"]
            ]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((rv - np.median(rv, axis=0)) / np.median(rv, axis=0))

        else:
            aie = obs[[column_mapping["a_au"],
                       column_mapping["i_deg"],
                       column_mapping["e"]]].values

            # Calculate the percent difference between the median of each velocity element
            # and the heliocentric distance
            percent_diff = np.abs((aie - np.median(aie, axis=0)) / np.median(aie, axis=0))


        # Sum the percent differences
        summed_diff = np.sum(percent_diff, axis=1)

        # Find the minimum summed percent difference and call that
        # the average object
        index = np.where(summed_diff == np.min(summed_diff))[0][0]
        name = obs[column_mapping["name"]].values[index]

        # Grab the objects, name and its r and v.
        obj_observations = obs[obs[column_mapping["name"]] == name]
        obj = obj_observations[[
            column_mapping["exp_mjd"],
            column_mapping["obj_x_au"],
            column_mapping["obj_y_au"],
            column_mapping["obj_z_au"],
            column_mapping["obj_dx/dt_au_p_day"],
            column_mapping["obj_dy/dt_au_p_day"],
            column_mapping["obj_dz/dt_au_p_day"],
            column_mapping["RA_deg"],
            column_mapping["Dec_deg"],
            column_mapping["r_au"],
            column_mapping["a_au"],
            column_mapping["i_deg"],
            column_mapping["e"],
            column_mapping["name"]]].copy()
        obj["orbit_id"] = i + 1

        average_orbits.append(obj[["orbit_id",
            column_mapping["exp_mjd"],
            column_mapping["obj_x_au"],
            column_mapping["obj_y_au"],
            column_mapping["obj_z_au"],
            column_mapping["obj_dx/dt_au_p_day"],
            column_mapping["obj_dy/dt_au_p_day"],
            column_mapping["obj_dz/dt_au_p_day"],
            column_mapping["RA_deg"],
            column_mapping["Dec_deg"],
            column_mapping["r_au"],
            column_mapping["a_au"],
            column_mapping["i_deg"],
            column_mapping["e"],
            column_mapping["name"]]])

    average_orbits = pd.concat(average_orbits)
    average_orbits.sort_values(by=["orbit_id", column_mapping["exp_mjd"]], inplace=True)
    average_orbits.reset_index(inplace=True, drop=True)

    return average_orbits
