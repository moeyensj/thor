import numpy as np

__all__ = ["calcLinkageEfficiency"]

def calcLinkageEfficiency(allObjects, 
                          vxRange=[-0.1, 0.1], 
                          vyRange=[-0.1, 0.1]
                          verbose=Config.verbose):
    """
    Calculates the linkage efficiency given the velocity ranges searched. 
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        The allObjects DataFrame. Requires analyzeProjections
        and analyzeClusters to have been run on the DataFrame. 
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x. 
        Will not be used if vxValues are specified. 
        [Default = [-0.1, 0.1]]
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y. 
        Will not be used if vyValues are specified. 
        [Default = [-0.1, 0.1]]
    
    Returns
    -------
    efficiency : float
        The number of objects found over the number findable as a fraction. 
    """
    # Find the objects that should have been found with the defined velocity ranges
    in_zone_findable = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["findable"] == 1))
    findable = len(allObjects[in_zone_findable])
    
    # Find the objects that were missed
    in_zone_missed = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 0)
         & (allObjects["findable"] == 1))
    missed = len(allObjects[in_zone_missed])
    
    # Find the objects that were found
    in_zone_found = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 1))
    found = len(allObjects[in_zone_found])
    
    efficiency = found / findable
    
    if verbose == True:
        print("Findable objects: {}".format(findable))
        print("Found objects: {}".format(found))
        print("Missed objects: {}".format(missed))
        print("Efficiency [%]: {.f2}".format(efficiency * 100))
    return efficiency