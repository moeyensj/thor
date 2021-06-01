import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic_2d

__all__ = [
    "plotBinnedContour",
    "plotScatterContour"
]

def plotBinnedContour(
        dataframe,
        x_column,
        y_column,
        z_column,
        statistic="median",
        log_statistic=False,
        plot_counts=False,
        log_counts=False,
        count_levels=10,
        bins=100,
        mask=None,
        x_label=None,
        y_label=None,
        z_label=None,
        contour_kwargs={
            "colors": "red",
            "linewidths": 1
        },
        imshow_kwargs={
            "aspect": "auto"
            }
        ):
    """
    Plots a binned 2D histogram with optional contours.

    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    x_column : str
        Name of column containing desired x values.
    y_column : str
        Name of column containing desired y values.
    z_column : str
        Name of column containing desired z values.
    statistic : str, optional
        The statistic to compute on values of z for bins in x and y.
        Uses scipy.stats.binned_statistic_2d.
        [Default = 'median']
    log_statistic : bool, optional
        Plot the log base 10 of the calculated statistic.
        [Defaults = False]
    plot_counts : bool, optional
        Plot contours of the counts in each bin of x and y.
        [Default = False]
    log_counts : bool, optional
        Make contours the log of the counts.
        [Default = False]
    count_levels : int, optional
        Plot this may contour levels.
        [Default = 10]
    bins : int, optional
        The number of bins in x and y to calculate the
        statistic. The total number of bins is bins*bins.
        [Default = 100]
    mask : {None, `~pandas.Series`}, optional
        A mask on dataframe that cuts the data to be plotted.
        [Default = None]
    x_label : {None, str}, optional
        If None, will set the x-axis label to x_column, if not None use
        this label instead.
        [Default = None]
    y_label : {None, str}, optional
        If None, will set the y-axis label to y_column, if not None use
        this label instead.
        [Default = None]
    z_label : {None, str}, optional
        If None, will set the colorbar label to z_column, if not None use
        this label instead.
        [Default = None]
    contour_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.contour.
        [Default = {'colors': 'red', 'linewidths': 1}]
    imshow_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.imshow.
        [Default = {'aspect' : 'auto'}]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """

    try:
        if mask.dtype == bool:
            dataframe = dataframe[mask]
    except:
        dataframe = dataframe

    X = binned_statistic_2d(
        dataframe[x_column].values,
        dataframe[y_column].values,
        dataframe[z_column].values,
        statistic=statistic,
        bins=bins
    )

    if log_statistic == True:
        stat = np.log10(X.statistic.T)
    else:
        stat = X.statistic.T

    fig, ax = plt.subplots(1, 1, dpi=600)
    cm = ax.imshow(stat,
                   origin="lower",
                   extent=[X.x_edge[0], X.x_edge[-1], X.y_edge[0], X.y_edge[-1]],
                   **imshow_kwargs)
    cb = fig.colorbar(cm)

    if z_label == None:
        cb.set_label("{} {}".format(statistic, z_column))
    else:
        cb.set_label(z_label)

    if plot_counts == True:
        N = binned_statistic_2d(
            dataframe[x_column].values,
            dataframe[y_column].values,
            dataframe[z_column].values,
            statistic="count",
            bins=bins)
        if log_counts == True:
            counts = np.log10(N.statistic.T)
        else:
            counts = N.statistic.T

        cs = ax.contour(counts,
                   count_levels,
                   origin="lower",
                   extent=[N.x_edge[0], N.x_edge[-1], N.y_edge[0], N.y_edge[-1]],
                   **contour_kwargs)
        plt.clabel(cs, inline=1, fontsize=5)

    if x_label == None:
        ax.set_x_label(x_column)
    else:
        ax.set_x_label(x_label)

    if y_label == None:
        ax.set_y_label(y_column)
    else:
        ax.set_y_label(y_label)

    return fig, ax

def plotScatterContour(
        dataframe,
        x_column,
        y_column,
        z_column,
        plot_counts=False,
        log_counts=False,
        count_levels=10,
        bins=100,
        mask=None,
        x_label=None,
        y_label=None,
        z_label=None,
        contour_kwargs={
            "colors": "red",
            "linewidths": 1
        },
        scatterKwargs={
            "s": 0.1
        }):
    """
    Plots a scatter plot with optional contours.

    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    x_column : str
        Name of column containing desired x values.
    y_column : str
        Name of column containing desired y values.
    z_column : str
        Name of column containing desired z values.
    plot_counts : bool, optional
        Plot contours of the counts in each bin of x and y.
        [Default = False]
    log_counts : bool, optional
        Make contours the log of the counts.
        [Default = False]
    count_levels : int, optional
        Plot this may contour levels.
        [Default = 10]
    bins : int, optional
        The number of bins in x and y to calculate the
        statistic. The total number of bins is bins*bins.
        [Default = 100]
    mask : {None, `~pandas.Series`}, optional
        A mask on dataframe that cuts the data to be plotted.
        [Default = None]
    x_label : {None, str}, optional
        If None, will set the x-axis label to x_column, if not None use
        this label instead.
        [Default = None]
    y_label : {None, str}, optional
        If None, will set the y-axis label to y_column, if not None use
        this label instead.
        [Default = None]
    z_label : {None, str}, optional
        If None, will set the colorbar label to z_column, if not None use
        this label instead.
        [Default = None]
    contour_kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.contour.
        [Default = {'colors': 'red', 'linewidths': 1}]
    scatterKwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.scatter.
        [Default = {'s': 0.1}]

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes object.
    """

    try:
        if mask.dtype == bool:
            dataframe = dataframe[mask]
    except:
        dataframe = dataframe

    fig, ax = plt.subplots(1, 1, dpi=600)
    cm = ax.scatter(
        dataframe[x_column].values,
        dataframe[y_column].values,
        c=dataframe[z_column].values,
        **scatterKwargs
    )
    cb = fig.colorbar(cm)

    if z_label == None:
        cb.set_label(z_column)
    else:
        cb.set_label(z_label)

    if plot_counts == True:
        N = binned_statistic_2d(
            dataframe[x_column].values,
            dataframe[y_column].values,
            dataframe[z_column].values,
            statistic="count",
            bins=bins)
        if log_counts == True:
            counts = np.log10(N.statistic.T)
        else:
            counts = N.statistic.T

        cs = ax.contour(counts,
                   count_levels,
                   origin="lower",
                   extent=[N.x_edge[0], N.x_edge[-1], N.y_edge[0], N.y_edge[-1]],
                   **contour_kwargs)
        plt.clabel(cs, inline=1, fontsize=5)

    if x_label == None:
        ax.set_x_label(x_column)
    else:
        ax.set_x_label(x_label)

    if y_label == None:
        ax.set_y_label(y_column)
    else:
        ax.set_y_label(y_label)

    return fig, ax