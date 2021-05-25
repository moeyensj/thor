import pandas as pd
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
              vx_range,
              vy_range):
    """
    Helper function that plots a rectangular shape.
    """
    rect = patches.Rectangle((vx_range[0], vy_range[0]),
                             vx_range[1]-vx_range[0],
                             vy_range[1]-vy_range[0],
                             linewidth=0.5,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    return

def plotProjectionVelocitiesFindable(
        all_truths,
        vx_range=None,
        vy_range=None
    ):
    """
    Plots objects that should be findable in the projection
    space based on their median velocities and the
    chosen velocity ranges.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'findable' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    if vx_range is not None and vy_range is not None:
        in_zone_findable = ((all_truths["dtheta_x/dt_median"] >= vx_range[0])
         & (all_truths["dtheta_x/dt_median"] <= vx_range[1])
         & (all_truths["dtheta_y/dt_median"] <= vy_range[1])
         & (all_truths["dtheta_y/dt_median"] >= vy_range[0])
         & (all_truths["findable"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(all_truths[all_truths["findable"] == 1]["dtheta_x/dt_median"].values,
                all_truths[all_truths["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=all_truths[all_truths["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=all_truths[all_truths["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    cm = ax.scatter(all_truths[all_truths["findable"] == 1]["dtheta_x/dt_median"].values,
                    all_truths[all_truths["findable"] == 1]["dtheta_y/dt_median"].values,
                    s=0.1,
                    c=all_truths[all_truths["findable"] == 1]["r_au_median"].values,
                    vmin=0,
                    vmax=5.0,
                    cmap="viridis")
    cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
    ax.set_aspect("equal")

    if vx_range is not None and vy_range is not None:
        _plotGrid(ax, vx_range, vy_range)

    # Add labels and text
    cb.set_label("r [AU]", size=10)
    ax.set_x_label(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_y_label(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04),
        _setPercentage(ax.get_ylim(), 0.05),
        "Objects Findable: {}".format(len(all_truths[all_truths["findable"] == 1])))

    if vx_range is not None and vy_range is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04),
                _setPercentage(ax.get_ylim(), 0.11),
                "Objects Findable in Grid: {}".format(len(all_truths[in_zone_findable])),
                color="r")
    return fig, ax

def plotProjectionVelocitiesFound(
        all_truths,
        vx_range=None,
        vy_range=None
    ):
    """
    Plots objects that were found in the projection
    space based on their median velocities and the
    chosen velocity ranges.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    if vx_range is not None and vy_range is not None:
        in_zone_found = ((all_truths["dtheta_x/dt_median"] >= vx_range[0])
         & (all_truths["dtheta_x/dt_median"] <= vx_range[1])
         & (all_truths["dtheta_y/dt_median"] <= vy_range[1])
         & (all_truths["dtheta_y/dt_median"] >= vy_range[0])
         & (all_truths["found"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(all_truths[all_truths["findable"] == 1]["dtheta_x/dt_median"].values,
                all_truths[all_truths["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=all_truths[all_truths["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=all_truths[all_truths["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    if len(all_truths[all_truths["found"] == 1]) != 0:
        cm = ax.scatter(all_truths[all_truths["found"] == 1]["dtheta_x/dt_median"].values,
                        all_truths[all_truths["found"] == 1]["dtheta_y/dt_median"].values,
                        s=0.1,
                        c=all_truths[all_truths["found"] == 1]["r_au_median"].values,
                        vmin=0,
                        vmax=5.0,
                        cmap="viridis")
        cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
        cb.set_label("r [AU]", size=10)
    ax.set_aspect("equal")

    if vx_range is not None and vy_range is not None:
        _plotGrid(ax, vx_range, vy_range)

    # Add labels and text
    ax.set_x_label(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_y_label(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04),
        _setPercentage(ax.get_ylim(), 0.05),
        "Objects Found: {}".format(len(all_truths[all_truths["found"] == 1])))

    if vx_range is not None and vy_range is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04),
                _setPercentage(ax.get_ylim(), 0.11),
                "Objects Found in Grid: {}".format(len(all_truths[in_zone_found])),
                color="g")
    return fig, ax

def plotProjectionVelocitiesMissed(
        all_truths,
        vx_range=None,
        vy_range=None
    ):
    """
    Plots objects that were missed in the projection
    space based on their median velocities and the
    chosen velocity ranges.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' and 'findable' column to be populated,
        as well as the median projection space velocities and the median
        heliocentric distance.
    vx_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in x.
        [Default = None]
    vy_range : {None, list or `~numpy.ndarray` (2)}
        Maximum and minimum velocity range in y.
        [Default = None]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    if vx_range is not None and vy_range is not None:
        in_zone_missed = ((all_truths["dtheta_x/dt_median"] >= vx_range[0])
            & (all_truths["dtheta_x/dt_median"] <= vx_range[1])
            & (all_truths["dtheta_y/dt_median"] <= vy_range[1])
            & (all_truths["dtheta_y/dt_median"] >= vy_range[0])
            & (all_truths["found"] == 0)
            & (all_truths["findable"] == 1))

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax.errorbar(all_truths[all_truths["findable"] == 1]["dtheta_x/dt_median"].values,
                all_truths[all_truths["findable"] == 1]["dtheta_y/dt_median"].values,
                yerr=all_truths[all_truths["findable"] == 1]["dtheta_y/dt_sigma"].values,
                xerr=all_truths[all_truths["findable"] == 1]["dtheta_x/dt_sigma"].values,
                fmt="o",
                ms=0.01,
                capsize=0.1,
                elinewidth=0.1,
                c="k", zorder=-1)
    if len(all_truths[(all_truths["findable"] == 1) & (all_truths["found"] == 0)]) > 0:
        cm = ax.scatter(all_truths[(all_truths["findable"] == 1) & (all_truths["found"] == 0)]["dtheta_x/dt_median"].values,
                        all_truths[(all_truths["findable"] == 1) & (all_truths["found"] == 0)]["dtheta_y/dt_median"].values,
                        s=0.1,
                        c=all_truths[(all_truths["findable"] == 1) & (all_truths["found"] == 0)]["r_au_median"].values,
                        vmin=0,
                        vmax=5.0,
                        cmap="viridis")
        cb = fig.colorbar(cm, fraction=0.02, pad=0.02)
        cb.set_label("r [AU]", size=10)
    ax.set_aspect("equal")

    if vx_range is not None and vy_range is not None:
        _plotGrid(ax, vx_range, vy_range)

    # Add labels and text
    ax.set_x_label(r"Median $ d\theta_X / dt$ [Degrees Per Day]", size=10)
    ax.set_y_label(r"Median $ d\theta_Y / dt$ [Degrees Per Day]", size=10)

    ax.text(_setPercentage(ax.get_xlim(), 0.04),
        _setPercentage(ax.get_ylim(), 0.05),
        "Objects Missed: {}".format(len(all_truths[(all_truths["findable"] == 1) & (all_truths["found"] == 0)])))

    if vx_range is not None and vy_range is not None:
        ax.text(_setPercentage(ax.get_xlim(), 0.04),
                _setPercentage(ax.get_ylim(), 0.11),
                "Objects Missed in Grid: {}".format(len(all_truths[in_zone_missed])),
                color="r")
    return fig, ax

def plotOrbitsFindable(
        all_truths,
        orbits,
        test_orbits=None,
        column_mapping=Config.COLUMN_MAPPING):
    """
    Plots orbits that should be findable in semi-major axis, inclination
    and eccentrity space.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'findable' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in
        the all_truths DataFrame.
    test_orbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red.
        [Default = None]
    column_mapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names.
        [Default = `~thor.Config.COLUMN_MAPPING`]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    findable = orbits[orbits[column_mapping["name"]].isin(all_truths[all_truths["findable"] == 1][column_mapping["name"]].values)]
    fig, ax = plotScatterContour(
        findable,
        column_mapping["a_au"],
        column_mapping["i_deg"],
        column_mapping["e"],
        plot_counts=False,
        log_counts=True,
        count_levels=4,
        mask=None,
        x_label="a [AU]",
        y_label="i [Degrees]",
        z_label="e",
        scatter_kwargs={
            "s": 0.1,
            "vmin": 0,
            "vmax": 1,
            "cmap":
            "viridis"
        }
    )
    if type(test_orbits) == pd.DataFrame:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[column_mapping["name"]].isin(test_orbits[column_mapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[column_mapping["a_au"], column_mapping["i_deg"]]].values.T,
                       s=2,
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
    return fig, ax


def plotOrbitsFound(
        all_truths,
        orbits,
        test_orbits=None,
        column_mapping=Config.COLUMN_MAPPING
    ):
    """
    Plots orbits that have been found in semi-major axis, inclination
    and eccentrity space.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in
        the all_truths DataFrame.
    test_orbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red.
        [Default = None]
    column_mapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names.
        [Default = `~thor.Config.COLUMN_MAPPING`]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    found = orbits[orbits[column_mapping["name"]].isin(all_truths[all_truths["found"] == 1][column_mapping["name"]].values)]
    fig, ax = plotScatterContour(
        found,
        column_mapping["a_au"],
        column_mapping["i_deg"],
        column_mapping["e"],
        plot_counts=False,
        log_counts=True,
        count_levels=4,
        mask=None,
        x_label="a [AU]",
        y_label="i [Degrees]",
        z_label="e",
        scatter_kwargs={"s": 0.1, "vmin": 0, "vmax": 1, "cmap": "viridis"}
    )
    if type(test_orbits) == pd.DataFrame:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[column_mapping["name"]].isin(test_orbits[column_mapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[column_mapping["a_au"], column_mapping["i_deg"]]].values.T,
                       s=2,
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')
    return fig, ax

def plotOrbitsMissed(all_truths,
                     orbits,
                     test_orbits=None,
                     column_mapping=Config.COLUMN_MAPPING):
    """
    Plots orbits that have been missed (but were findable) in semi-major axis, inclination
    and eccentrity space.

    Parameters
    ----------
    all_truths : `~pandas.DataFrame`
        Object summary DataFrame. Needs 'found' and 'findable' column to be populated.
    orbits : `~pandas.DataFrame`
        Orbit DataFrame, should contain the orbits for the objects in
        the all_truths DataFrame.
    test_orbits : {None, `~pandas.DataFrame`}, optional
        If passed, will plot test orbits in red.
        [Default = None]
    column_mapping : dict, optional
        Column name mapping of orbits DataFrame to internally used column names.
        [Default = `~thor.Config.COLUMN_MAPPING`]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """
    missed = orbits[orbits[column_mapping["name"]].isin(all_truths[(all_truths["found"] == 0) & (all_truths["findable"] == 1)][column_mapping["name"]].values)]
    fig, ax = plotScatterContour(
        missed,
        column_mapping["a_au"],
        column_mapping["i_deg"],
        column_mapping["e"],
        plot_counts=False,
        log_counts=True,
        count_levels=4,
        mask=None,
        x_label="a [AU]",
        y_label="i [Degrees]",
        z_label="e",
        scatter_kwargs={
            "s": 0.1,
            "vmin": 0,
            "vmax": 1,
            "cmap": "viridis"
        }
    )
    if type(test_orbits) == pd.DataFrame:
        # If test orbits exist in known orbits, plot them
        test_orbits_in_known = orbits[orbits[column_mapping["name"]].isin(test_orbits[column_mapping["name"]].values)]
        if len(test_orbits_in_known) != 0:
            ax.scatter(*test_orbits_in_known[[column_mapping["a_au"], column_mapping["i_deg"]]].values.T,
                       s=2,
                       c="r",
                       label="Test Orbits")
            ax.legend(loc='upper right')

    return fig, ax