import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic_2d

__all__ = ["plotBinnedContour",
           "plotScatterContour"]

def plotBinnedContour(dataframe, 
                      xColumn,
                      yColumn, 
                      zColumn, 
                      statistic="median", 
                      logStatistic=False, 
                      plotCounts=False, 
                      logCounts=False, 
                      countLevels=10, 
                      bins=100, 
                      mask=None,
                      xLabel=None,
                      yLabel=None,
                      zLabel=None,
                      contourKwargs={"colors": "red",
                                      "linewidths": 1},
                      imshowKwargs={"aspect": "auto"}):
    """
    Plots a binned 2D histogram with optional contours. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    xColumn : str
        Name of column containing desired x values.
    yColumn : str
        Name of column containing desired y values.
    zColumn : str
        Name of column containing desired z values.
    statistic : str, optional
        The statistic to compute on values of z for bins in x and y. 
        Uses scipy.stats.binned_statistic_2d.
        [Default = 'median']
    logStatistic : bool, optional
        Plot the log base 10 of the calculated statistic.
        [Defaults = False]
    plotCounts : bool, optional
        Plot contours of the counts in each bin of x and y. 
        [Default = False]
    logCounts : bool, optional
        Make contours the log of the counts.
        [Default = False]
    countLevels : int, optional
        Plot this may contour levels. 
        [Default = 10]
    bins : int, optional
        The number of bins in x and y to calculate the
        statistic. The total number of bins is bins*bins.
        [Default = 100]
    mask : {None, `~pandas.Series`}, optional
        A mask on dataframe that cuts the data to be plotted.
        [Default = None]
    xLabel : {None, str}, optional
        If None, will set the x-axis label to xColumn, if not None use
        this label instead. 
        [Default = None]
    yLabel : {None, str}, optional
        If None, will set the y-axis label to yColumn, if not None use
        this label instead. 
        [Default = None]
    zLabel : {None, str}, optional
        If None, will set the colorbar label to zColumn, if not None use
        this label instead. 
        [Default = None]
    contourKwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ax.contour. 
        [Default = {'colors': 'red', 'linewidths': 1}]
    imshowKwargs : dict, optional
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
        dataframe[xColumn].values, 
        dataframe[yColumn].values, 
        dataframe[zColumn].values, 
        statistic=statistic,
        bins=bins)
    
    if logStatistic == True:
        stat = np.log10(X.statistic.T)
    else:
        stat = X.statistic.T
    
    fig, ax = plt.subplots(1, 1, dpi=200)
    cm = ax.imshow(stat, 
                   origin="lower", 
                   extent=[X.x_edge[0], X.x_edge[-1], X.y_edge[0], X.y_edge[-1]], 
                   **imshowKwargs)
    cb = fig.colorbar(cm)
    
    if zLabel == None:
        cb.set_label("{} {}".format(statistic, zColumn))
    else:
        cb.set_label(zLabel)

    if plotCounts == True:
        N = binned_statistic_2d(
            dataframe[xColumn].values, 
            dataframe[yColumn].values, 
            dataframe[zColumn].values, 
            statistic="count",
            bins=bins)
        if logCounts == True:
            counts = np.log10(N.statistic.T)
        else:
            counts = N.statistic.T
        
        cs = ax.contour(counts, 
                   countLevels, 
                   origin="lower", 
                   extent=[N.x_edge[0], N.x_edge[-1], N.y_edge[0], N.y_edge[-1]],
                   **contourKwargs)
        plt.clabel(cs, inline=1, fontsize=5)
    
    if xLabel == None:
        ax.set_xlabel(xColumn)
    else:
        ax.set_xlabel(xLabel)
    
    if yLabel == None:
        ax.set_ylabel(yColumn)
    else:
        ax.set_ylabel(yLabel)
    
    return fig, ax

def plotScatterContour(dataframe, 
                      xColumn,
                      yColumn, 
                      zColumn,
                      plotCounts=False, 
                      logCounts=False, 
                      countLevels=10, 
                      bins=100, 
                      mask=None,
                      xLabel=None,
                      yLabel=None,
                      zLabel=None,
                      contourKwargs={"colors": "red",
                                    "linewidths": 1},
                      scatterKwargs={"s": 0.1}):
    """
    Plots a scatter plot with optional contours. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    xColumn : str
        Name of column containing desired x values.
    yColumn : str
        Name of column containing desired y values.
    zColumn : str
        Name of column containing desired z values.
    plotCounts : bool, optional
        Plot contours of the counts in each bin of x and y. 
        [Default = False]
    logCounts : bool, optional
        Make contours the log of the counts.
        [Default = False]
    countLevels : int, optional
        Plot this may contour levels. 
        [Default = 10]
    bins : int, optional
        The number of bins in x and y to calculate the
        statistic. The total number of bins is bins*bins.
        [Default = 100]
    mask : {None, `~pandas.Series`}, optional
        A mask on dataframe that cuts the data to be plotted.
        [Default = None]
    xLabel : {None, str}, optional
        If None, will set the x-axis label to xColumn, if not None use
        this label instead. 
        [Default = None]
    yLabel : {None, str}, optional
        If None, will set the y-axis label to yColumn, if not None use
        this label instead. 
        [Default = None]
    zLabel : {None, str}, optional
        If None, will set the colorbar label to zColumn, if not None use
        this label instead. 
        [Default = None]
    contourKwargs : dict, optional
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
    
    fig, ax = plt.subplots(1, 1, dpi=200)
    cm = ax.scatter(dataframe[xColumn].values, 
                    dataframe[yColumn].values, 
                    c=dataframe[zColumn].values, 
                    **scatterKwargs)
    cb = fig.colorbar(cm)
    
    if zLabel == None:
        cb.set_label(zColumn)
    else:
        cb.set_label(zLabel)

    if plotCounts == True:
        N = binned_statistic_2d(
            dataframe[xColumn].values, 
            dataframe[yColumn].values, 
            dataframe[zColumn].values, 
            statistic="count",
            bins=bins)
        if logCounts == True:
            counts = np.log10(N.statistic.T)
        else:
            counts = N.statistic.T
        
        cs = ax.contour(counts, 
                   countLevels, 
                   origin="lower", 
                   extent=[N.x_edge[0], N.x_edge[-1], N.y_edge[0], N.y_edge[-1]],
                   **contourKwargs)
        plt.clabel(cs, inline=1, fontsize=5)
    
    if xLabel == None:
        ax.set_xlabel(xColumn)
    else:
        ax.set_xlabel(xLabel)
    
    if yLabel == None:
        ax.set_ylabel(yColumn)
    else:
        ax.set_ylabel(yLabel)
    
    return fig, ax