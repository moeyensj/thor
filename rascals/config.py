__all__ = ["Config"]

class Config:
    
    observationColumns = {"obs_id" : "obsId",
                          "exp_mjd" : "mjd_utc",
                          "night": "night",
                          "RA_deg" : "RA_deg",
                          "Dec_deg" : "Dec_deg",
                          "obs_x_au" : "HEclObsy_X_au",
                          "obs_y_au" : "HEclObsy_Y_au",
                          "obs_z_au" : "HEclObsy_Z_au",
                          "mag" : "VMag"}
    
    truthColumns = {"name" : "designation",
                    "Delta_au" : "Delta_au",
                    "r_au" : "r_au",
                    "obj_x_au" :"HEclObj_X_au",
                    "obj_y_au" : "HEclObj_Y_au",
                    "obj_z_au" : "HEclObj_Z_au",
                    "obj_dx/dt_au_p_day" : "HEclObj_dX/dt_au_p_day",
                    "obj_dy/dt_au_p_day" : "HEclObj_dY/dt_au_p_day",
                    "obj_dz/dt_au_p_day" : "HEclObj_dZ/dt_au_p_day"}
    
    # MUST BE SET ACCORDINGLY
    ### TODO: have code search for oorb directory
    ###       use a pre-defined configuration file for oorb
    oorbDirectory = "/Users/joachim/repos/cloned/oorb/" 
    oorbObservatoryCode = "I11"
    oorbConfigFile ="/Users/joachim/repos/RaSCaLS/rascals/data/oorb.conf"
