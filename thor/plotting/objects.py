import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from .helpers import _setAxes

__all__ = ["plotCell"]

def plotCell(cell,
             coordinateSystem="equatorialAngular",
             scatterKwargs={"s":0.05},
             columnMapping=Config.columnMapping):
    """
    Plot cell. Needs cell's observations to be loaded.
    
    Parameters
    ----------
    cell : `~thor.cell.Cell`
        THOR cell. 
    coordinateSystem : {'equatorialAngular', 'eclipticAngular'}, optional
        Which coordinate system to use.
        [Default = 'equatorialAngular']
    scatterKwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.scatter.
        [Default = {'s': 0.05}]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.columnMapping`]
        
    Returns
    -------
    fig : `~matplotlib.figure.Figure` 
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object. 
    """
        
    fig, ax = plt.subplots(1, 1, dpi=200)
    fig.tight_layout()
    ax.set_aspect("equal")

    if coordinateSystem == "equatorialAngular":
        x = cell.observations[columnMapping["RA_deg"]].values, 
        y = cell.observations[columnMapping["Dec_deg"]].values
    elif coordinateSystem == "eclipticAngular":
        x = cell.observations[columnMapping["lon_deg"]].values, 
        y = cell.observations[columnMapping["lat_deg"]].values
    else:
        raise ValueError("coordinateSystem should be one of 'equatorialAngular' or 'eclipticAngular'")
    
    _setAxes(ax, coordinateSystem)
    ax.scatter(x, y, **scatterKwargs)
    
    if cell.shape == "circle":
        cell_p = plt.Circle((cell.center[0], cell.center[1]), np.sqrt(cell.area / np.pi), color="r", fill=False)
    elif cell.shape == "square":
        half_side = np.sqrt(cell.area) / 2
        cell_p = plt.Rectangle((cell.center[0] - half_side, cell.center[1] - half_side), 2 * half_side, 2 * half_side, color="r", fill=False)
    else: 
        raise ValueError("Cell.shape should be one of 'square' or 'circle'")
    ax.add_artist(cell_p)
    ax.grid()
    return fig, ax