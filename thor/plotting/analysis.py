import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..config import Config
from .helpers import _setPercentage
from .contour import plotScatterContour

__all__ = ["_plotGrid",
           "plotProjectionVelocitiesFindable",
           "plotProjectionVelocitiesFound",
           "plotProjectionVelocitiesMissed",
           "plotOrbitsFindable",
           "plotOrbitsFound",
           "plotOrbitsMissed"]

def _plotGrid(ax, 
              vxRange, 
              vyRange):
    """
    Helper function that plots a rectangular shape.
    """
    rect = patches.Rectangle((vxRange[0], vyRange[0]),
                             vxRange[1]-vxRange[0],
                             vyRange[1]-vyRange[0], 
                             linewidth=0.5,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    return

def plotProjectionVelocitiesFindable(allObjects,
                                     vxRange=None,
                                     vyRange=None):
    """
    Plots objects that should be findable in the projection
    space based on their median velocities and the
    chosen velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'findable' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vyRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    if vxRange is not None and vyRange is not None:
        in_zone_findable = ((allObjects["dtheta_x/dt_median"] >= vxRange[0])
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1])
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1])
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["findable"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(allObjects[allObjects["findable"] == 1]["dtheta_x/dt_median"].values,
                allObjects[allObjects["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=allObjects[allObjects["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=allObjects[allObjects["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    cm = ax.scatter(allObjects[allObjects["findable"] == 1]["dtheta_x/dt_median"].values, 
                    allObjects[allObjects["findable"] == 1]["dtheta_y/dt_median"].values,
                    s=0.1,
                    c=allObjects[allObjects["findable"] == 1]["r_au_median"].values,
                    vmin=0,
                    vmax=5.0,
                    cmap="viridis")
    cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
    ax.set_aspect("equal")

    if vxRange is not None and vyRange is not None:
        _plotGrid(ax, vxRange, vyRange)

    # Add labels and text
    cb.set_label("r [AU]", size=10)
    ax.set_xlabel(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_ylabel(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04), 
        _setPercentage(ax.get_ylim(), 0.05), 
        "Objects Findable: {}".format(len(allObjects[allObjects["findable"] == 1])))

    if vxRange is not None and vyRange is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04), 
                _setPercentage(ax.get_ylim(), 0.11), 
                "Objects Findable in Grid: {}".format(len(allObjects[in_zone_findable])),
                color="r")
    return fig, ax

def plotProjectionVelocitiesFound(allObjects,
                                  vxRange=None,
                                  vyRange=None):
    """
    Plots objects that were found in the projection
    space based on their median velocities and the
    chosen velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vyRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    if vxRange is not None and vyRange is not None:
        in_zone_found = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(allObjects[allObjects["findable"] == 1]["dtheta_x/dt_median"].values,
                allObjects[allObjects["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=allObjects[allObjects["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=allObjects[allObjects["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    if len(allObjects[allObjects["found"] == 1]) != 0:
        cm = ax.scatter(allObjects[allObjects["found"] == 1]["dtheta_x/dt_median"].values, 
                        allObjects[allObjects["found"] == 1]["dtheta_y/dt_median"].values,
                        s=0.1,
                        c=allObjects[allObjects["found"] == 1]["r_au_median"].values,
                        vmin=0,
                        vmax=5.0,
                        cmap="viridis")
        cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
        cb.set_label("r [AU]", size=10)
    ax.set_aspect("equal")

    if vxRange is not None and vyRange is not None:
        _plotGrid(ax, vxRange, vyRange)

    # Add labels and text
    ax.set_xlabel(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_ylabel(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04), 
        _setPercentage(ax.get_ylim(), 0.05), 
        "Objects Found: {}".format(len(allObjects[allObjects["found"] == 1])))

    if vxRange is not None and vyRange is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04), 
                _setPercentage(ax.get_ylim(), 0.11), 
                "Objects Found in Grid: {}".format(len(allObjects[in_zone_found])), 
                color="g")
    return fig, ax

def plotProjectionVelocitiesMissed(allObjects,
                                  vxRange=None,
                                  vyRange=None):
    """
    Plots objects that were missed in the projection
    space based on their median velocities and the
    chosen velocity ranges.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' and 'findable' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vxRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vyRange : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    if vxRange is not None and vyRange is not None:
        in_zone_missed = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
            & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
            & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
            & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
            & (allObjects["found"] == 0)
            & (allObjects["findable"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(allObjects[allObjects["findable"] == 1]["dtheta_x/dt_median"].values,
                allObjects[allObjects["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=allObjects[allObjects["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=allObjects[allObjects["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    cm = ax.scatter(allObjects[(allObjects["findable"] == 1) & (allObjects["found"] == 0)]["dtheta_x/dt_median"].values,
                    allObjects[(allObjects["findable"] == 1) & (allObjects["found"] == 0)]["dtheta_y/dt_median"].values,
                    s=0.1,
                    c=allObjects[(allObjects["findable"] == 1) & (allObjects["found"] == 0)]["r_au_median"].values,
                    vmin=0,
                    vmax=5.0,
                    cmap="viridis")
    cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
    ax.set_aspect("equal")

    if vxRange is not None and vyRange is not None:
        _plotGrid(ax, vxRange, vyRange)

    # Add labels and text
    cb.set_label("r [AU]", size=10)
    ax.set_xlabel(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_ylabel(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04), 
        _setPercentage(ax.get_ylim(), 0.05), 
        "Objects Missed: {}".format(len(allObjects[(allObjects["findable"] == 1) & (allObjects["found"] == 0)])))

    if vxRange is not None and vyRange is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04), 
                _setPercentage(ax.get_ylim(), 0.11), 
                "Objects Missed in Grid: {}".format(len(allObjects[in_zone_missed])),
                color="r")
    return fig, ax

def plotOrbitsFindable(allObjects, 
                       orbits, 
                       testOrbits=None,
                       columnMapping=Config.columnMapping):
    """
    Plots orbits that should be findable in semi-major axis, inclination 
    and eccentrity space.

    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'findable' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in 
        the allObjects DataFrame.
    testOrbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red. 
        [Default = None]
    columnMapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    findable = orbits[orbits[columnMapping["name"]].isin(allObjects[allObjects["findable"] == 1][columnMapping["name"]].values)]
    fig, ax = plotScatterContour(findable, 
                                 columnMapping["a_au"],
                                 columnMapping["i_deg"],
                                 columnMapping["e"],
                                 plotCounts=False, 
                                 logCounts=True, 
                                 countLevels=4, 
                                 mask=None,
                                 xLabel="a [AU]",
                                 yLabel="i [Degrees]",
                                 zLabel="e",
                                 scatterKwargs={"s": 0.1, "vmin": 0, "vmax": 1, "cmap": "viridis"})
    if type(testOrbits) != None:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[columnMapping["name"]].isin(testOrbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
    return fig, ax
    
    
def plotOrbitsFound(allObjects, 
                    orbits, 
                    testOrbits=None,
                    columnMapping=Config.columnMapping):
    """
    Plots orbits that have been found in semi-major axis, inclination 
    and eccentrity space.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in 
        the allObjects DataFrame.
    testOrbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red. 
        [Default = None]
    columnMapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    found = orbits[orbits[columnMapping["name"]].isin(allObjects[allObjects["found"] == 1][columnMapping["name"]].values)]
    fig, ax = plotScatterContour(found, 
                                 columnMapping["a_au"],
                                 columnMapping["i_deg"],
                                 columnMapping["e"],
                                 plotCounts=False, 
                                 logCounts=True, 
                                 countLevels=4, 
                                 mask=None,
                                 xLabel="a [AU]",
                                 yLabel="i [Degrees]",
                                 zLabel="e",
                                 scatterKwargs={"s": 0.1, "vmin": 0, "vmax": 1, "cmap": "viridis"})
    if type(testOrbits) != None:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[columnMapping["name"]].isin(testOrbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
    return fig, ax
    
def plotOrbitsMissed(allObjects, 
                     orbits, 
                     testOrbits=None, 
                     columnMapping=Config.columnMapping):
    """
    Plots orbits that have been missed (but were findable) in semi-major axis, inclination 
    and eccentrity space.
    
    Parameters
    ----------
    allObjects : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' and 'findable' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in 
        the allObjects DataFrame.
    testOrbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red. 
        [Default = None]
    columnMapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
    missed = orbits[orbits[columnMapping["name"]].isin(allObjects[(allObjects["found"] == 0) & (allObjects["findable"] == 1)][columnMapping["name"]].values)]
    fig, ax = plotScatterContour(missed, 
                                 columnMapping["a_au"],
                                 columnMapping["i_deg"],
                                 columnMapping["e"],
                                 plotCounts=False, 
                                 logCounts=True, 
                                 countLevels=4, 
                                 mask=None,
                                 xLabel="a [AU]",
                                 yLabel="i [Degrees]",
                                 zLabel="e",
                                 scatterKwargs={"s": 0.1, "vmin": 0, "vmax": 1, "cmap": "viridis"})
    if type(testOrbits) != None:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[columnMapping["name"]].isin(testOrbits[columnMapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[columnMapping["a_au"], columnMapping["i_deg"]]].values.T, 
                       s=2, 
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')

    return fig, ax