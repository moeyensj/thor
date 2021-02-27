import numpy as np

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
    MIN_OBS = 5
    CONTAMINATION_PERCENTAGE = 20
    BACKEND = "PYOORB"
    BACKEND_KWARGS = {}
    NUM_THREADS = 60

    USE_RAY = False
    USE_GPU = False

    RANGE_SHIFT_CONFIG = {
        "cell_area" : 1000, 
        "threads" : NUM_THREADS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
    }

    CLUSTER_LINK_CONFIG = {
        "vx_range" : [-0.1, 0.1], 
        "vy_range" : [-0.1, 0.1],
        "vx_bins" : 300, 
        "vy_bins" : 300,
        "vx_values" : None,
        "vy_values" : None,
        "eps" : 5/3600, 
        "min_samples" : MIN_OBS,
        "threads" : NUM_THREADS
    }

    IOD_CONFIG = {
        "min_obs" : MIN_OBS,
        "contamination_percentage" : CONTAMINATION_PERCENTAGE,
        "rchi2_threshold" : 1000,
        "observation_selection_method" : "combinations",
        "iterate" : False,
        "light_time" : True,
        "linkage_id_col" : "cluster_id",
        "identify_subsets" : True,
        "threads" : NUM_THREADS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
    }

    OD_CONFIG = {
        "min_obs" : MIN_OBS,
        "contamination_percentage" : CONTAMINATION_PERCENTAGE,
        "rchi2_threshold" : 100,
        "delta" : 1e-6,
        "max_iter" : 10,
        "method" : "central",
        "fit_epoch" : False,
        "test_orbit" : None,
        "threads" : NUM_THREADS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
    }

    ODP_CONFIG = {
        "min_obs" : MIN_OBS,
        "contamination_percentage" : CONTAMINATION_PERCENTAGE,
        "rchi2_threshold" : 100,
        "eps" : 1/3600, 
        "delta" : 1e-6,
        "max_iter" : 10,
        "method" : "central",
        "fit_epoch" : False,
        "orbits_chunk_size" : 1,
        "observations_chunk_size" : 100000,
        "threads" : NUM_THREADS,
        "backend" : BACKEND,
        "backend_kwargs" : BACKEND_KWARGS,
    }

    ADES_METADATA = {
        "observatory_code" : "I11",
        "observatory_name" : "Vera C. Rubin Observatory",
        "telescope_aperture" : "8.4",
        "telescope_design" : "Reflector",
        "telescope_detector" : "CCD",
        "submitter" : "D. iRAC",
        "observers" : ["D. iRAC"],
        "measurers" : ["D. iRAC"],
    }
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