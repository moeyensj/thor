from .data_processing import findObsInCell

__all__ = ["Cell"]

class Cell:
    
    def __init__(self, center, mjd, shape="square", area=10, dataframe=None):
        
        self.center = center
        self.area = area
        self.shape = shape
        self.mjd = mjd
        #self.mjd_range = mjd_range
        self.dataframe = dataframe
        self.observations = None
        
    def getObservations(self):
        """
        
        
        """
        if self.dataframe is None:
            raise ValueError("Cell has no associated dataframe. Please redefine cell and add observations.")
        observations = self.dataframe
        nightly_observations = observations[(observations["exp_mjd"] >= self.mjd)
                                             & (observations["exp_mjd"] <= self.mjd + 0.00001)]
        keep = findObsInCell(nightly_observations["obs_id"].values,
                             nightly_observations[["RA_deg", "Dec_deg"]].values,
                             self.center,
                             fieldArea=self.area,
                             fieldShape=self.shape)
        self.observations = nightly_observations[nightly_observations["obs_id"].isin(keep)]
 