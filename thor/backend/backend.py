__all__ = ["Backend"]

class Backend:
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.is_setup = False
        return  
    
    def setup(self):
        return
    
    def propagateOrbits(self, orbits, t0, t1):
        """
        Propagate orbits from t0 to t1. 
        
        """
        err = (
            "This backend does not have orbit propagation implemented."
        )
        raise NotImplementedError(err)
    
    def generateEphemeris(self, orbits, t0, observers):
        """
        Generate ephemerides for the given orbits as observed by 
        the observers.
        
        """
        err = (
            "This backend does not have ephemeris generation implemented."
        )
        raise NotImplementedError(err)
        
    def orbitDeterminaton(self):
        err = (
            "This backend does not have orbit determination implemented."
        )
        raise NotImplementedError(err)
    