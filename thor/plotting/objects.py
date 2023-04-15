import matplotlib.pyplot as plt
import numpy as np

from .helpers import _setAxes

__all__ = ["plotCell"]


def plotCell(cell, coordinate_system="equatorial_angular", scatter_kwargs={"s": 0.05}):
    """
    Plot cell. Needs cell's observations to be loaded.

    Parameters
    ----------
    cell : `~thor.cell.Cell`
        THOR cell.
    coordinate_system : {'equatorial_angular', 'ecliptic_angular'}, optional
        Which coordinate system to use.
        [Default = 'equatorial_angular']
    scatter_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.scatter.
        [Default = {'s': 0.05}]

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

    if coordinate_system == "equatorial_angular":
        x = (cell.observations["RA_deg"].values,)
        y = cell.observations["Dec_deg"].values
    elif coordinate_system == "ecliptic_angular":
        x = (cell.observations["lon_deg"].values,)
        y = cell.observations["lat_deg"].values
    else:
        raise ValueError(
            "coordinate_system should be one of 'equatorial_angular' or 'ecliptic_angular'"
        )

    _setAxes(ax, coordinate_system)
    ax.scatter(x, y, **scatter_kwargs)

    if cell.shape == "circle":
        cell_p = plt.Circle(
            (cell.center[0], cell.center[1]),
            np.sqrt(cell.area / np.pi),
            color="r",
            fill=False,
        )
    elif cell.shape == "square":
        half_side = np.sqrt(cell.area) / 2
        cell_p = plt.Rectangle(
            (cell.center[0] - half_side, cell.center[1] - half_side),
            2 * half_side,
            2 * half_side,
            color="r",
            fill=False,
        )
    else:
        raise ValueError("Cell.shape should be one of 'square' or 'circle'")
    ax.add_artist(cell_p)
    ax.grid()
    return fig, ax
