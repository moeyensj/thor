__all__ = ["Config"]

class Config:
    objectColumns = {"name": "designation"}
    observationColumns = {"obsId": "obsId",
                          "mjd": "mjd_utc",
                          "night": "night",
                          "obs_x": "HEclObsy_X_au",
                          "obs_y": "HEclObsy_Y_au",
                          "obs_z": "HEclObsy_Z_au",
                          "alt": "VMag",
                          "alpha": "PhaseAngle_deg"}
    primaryCoordinateSystem = "equatorialAngularTopo"
    additionalCoordinateSystems = ["eclipticAngularHelio", "eclipticCartesianHelio"]
    coordinateColumns = {"equatorialAngularTopo": ["RA_deg",
                                                   "Dec_deg",
                                                   "Delta_au"],
                         "eclipticAngularHelio": ["HLon_deg",
                                                  "HLat_deg",
                                                  "r_au"],
                         "eclipticCartesianHelio": ["HEclObj_X_au",
                                                    "HEclObj_Y_au",
                                                    "HEclObj_Z_au"]}
    velocityColumns = {"equatorialAngularTopo": ["dRA/dt_deg_p_day",
                                                 "dDec/dt_deg_p_day",
                                                 "dDelta/dt_au_p_day"],
                       "eclipticCartesianHelio": ["HEclObj_dX/dt_au_p_day",
                                                  "HEclObj_dY/dt_au_p_day",
                                                  "HEclObj_dZ/dt_au_p_day"]}
    # MUST BE SET ACCORDINGLY
    ### TODO: have code search for oorb directory
    ###       use a pre-defined configuration file for oorb
    oorbDirectory = "/Users/joachim/repos/cloned/oorb/" 
    oorbObservatoryCode = "I11"
