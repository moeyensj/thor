import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

__all__ = ["_plotGrid",
           "plotProjectionAnalysis",
           "plotClusteringAnalysis"]

def _plotGrid(ax, vxRange, vyRange):
    rect = patches.Rectangle((vxRange[0], vyRange[0]),
                             vxRange[1]-vxRange[0],
                             vyRange[1]-vyRange[0], linewidth=0.5, edgecolor='r',facecolor='none')
    ax.add_patch(rect)

def plotProjectionAnalysis(allObjects, vxRange=[-0.1, 0.1], vyRange=[-0.1, 0.1], title="Findable Objects"):
    
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
               c=allObjects[allObjects["findable"] == 1]["r_au_median"].values,  vmin=0, vmax=5.0)
    cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
    ax.set_aspect("equal")
    _plotGrid(ax, vxRange, vyRange)

    # Add labels and text
    cb.set_label("r [AU]", size=10)
    ax.set_xlabel(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_ylabel(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)
    ax.set_title(title)
    ax.text(-0.38, -0.15, "Objects: {}".format(len(allObjects[allObjects["findable"] == 1])))
    ax.text(-0.38, -0.18, "Objects Findable: {}".format(len(allObjects[in_zone_findable])), color="r")
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.2, 0.2)
    return fig, ax

def plotClusteringAnalysis(allObjects,  vxRange=[-0.1, 0.1], vyRange=[-0.1, 0.1], title="Missed Objects"):
   
    in_zone_findable = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["findable"] == 1))
    
    in_zone_missed = ((allObjects["dtheta_x/dt_median"] >= vxRange[0]) 
         & (allObjects["dtheta_x/dt_median"] <= vxRange[1]) 
         & (allObjects["dtheta_y/dt_median"] <= vyRange[1]) 
         & (allObjects["dtheta_y/dt_median"] >= vyRange[0])
         & (allObjects["found"] == 0)
         & (allObjects["findable"] == 1))
    
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
                c="k", zorder=-1, alpha=0.5)
    cm = ax.scatter(allObjects[in_zone_found]["dtheta_x/dt_median"].values, 
               allObjects[in_zone_found]["dtheta_y/dt_median"].values,
               s=0.1,
               c=allObjects[in_zone_found]["r_au_median"].values,  vmin=0, vmax=5.0)
    cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
    ax.set_aspect("equal")
    _plotGrid(ax, vxRange, vyRange)

    # Add labels and text
    cb.set_label("r [AU]", size=10)
    ax.set_xlabel(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_ylabel(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)
    ax.set_title(title)
    ax.text(-0.38, -0.15, "Objects: {}".format(len(allObjects[allObjects["findable"] == 1])))
    ax.text(-0.38, -0.18, "Objects Found: {}".format(len(allObjects[in_zone_found])), color="g")
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.2, 0.2)
    return fig, ax