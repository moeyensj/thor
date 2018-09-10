__all__ = ["Config"]

class Config:
    """
    Config: Holds configuration settings. 
    
    Of interest to the user are two main attributes:
        columnMapping : This dictionary should define the data
            column names of the user's data relative to the 
            internally used names.
        oorbDirectory : Oorb install location should be defined 
            here. 
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """

    columnMapping = {        
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
    
    observationColumns = [
        "obs_id",
        "exp_mjd",
        "visit_id",
        "field_id",
        "field_RA_deg",
        "field_Dec_deg",
        "night",
        "RA_deg",
        "Dec_deg", 
        "obs_x_au", 
        "obs_y_au", 
        "obs_z_au", 
        "mag",
    ]
    
    truthColumns = [
        "name", 
        "Delta_au",
        "r_au",
        "obj_x_au", 
        "obj_y_au", 
        "obj_z_au",
        "obj_dx/dt_au_p_day",
        "obj_dy/dt_au_p_day",
        "obj_dz/dt_au_p_day",
    ]
    
    # Convenience arrays
    x_e = [
        columnMapping["obs_x_au"], 
        columnMapping["obs_y_au"], 
        columnMapping["obs_z_au"]
    ]
    
    x_a = [
        columnMapping["obj_x_au"], 
        columnMapping["obj_y_au"], 
        columnMapping["obj_z_au"]
    ]
    
    v = [
        columnMapping["obj_dx/dt_au_p_day"], 
        columnMapping["obj_dy/dt_au_p_day"], 
        columnMapping["obj_dz/dt_au_p_day"]
    ]
    
    radec = [
        columnMapping["RA_deg"],
        columnMapping["Dec_deg"]
    ]
    
    thetaxy = [
        "theta_x_deg",
        "theta_y_deg"
    ]
    
    # MUST BE SET ACCORDINGLY
    ### TODO: have code search for oorb directory
    ###       use a pre-defined configuration file for oorb
    oorbDirectory = "/Users/joachim/repos/cloned/oorb/" 
    oorbObservatoryCode = "I11"
    oorbConfigFile ="/Users/joachim/repos/thor/thor/data/oorb.conf"
