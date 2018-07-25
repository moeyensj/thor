import plotly
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from .helpers import _setAxes

__all__ = ["plotObservations",
           "plotObservations3D"]

def plotObservations(dataframe, 
                    colorByObject=False, 
                    usePlotly=True, 
                    returnFig=False, 
                    columnMapping=Config.columnMapping):
    """
    Plot observations in 2D. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    colorByObject : bool, optional
        Color each unique object separately. 
        [Default = False]
    usePlotly : bool, optional
        Use plotly instead of matplotlib?
        [Default = True]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~rascals.Config.columnMapping`]
   
    Returns
    -------
    fig : {`~matplotlib.figure.Figure`, `~plotly.figure`}
        Returns the matplotlib or plotly figure object depending
        on the selected plotting package.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        If plotting with matplotlib, also returns the axis object.
    """
    if usePlotly is True:
        data = []
        if colorByObject is True:
            for name in dataframe[columnMapping["name"]].unique():
                obj = dataframe[dataframe[columnMapping["name"]] == name]
                if name == "NS":
                    trace = plotly.graph_objs.Scatter(
                        x=obj[columnMapping["RA_deg"]],
                        y=obj[columnMapping["Dec_deg"]],
                        name=name,
                        mode="markers",
                        marker=dict(size=2))
                else:
                    trace = plotly.graph_objs.Scatter(
                        x=obj[columnMapping["RA_deg"]],
                        y=obj[columnMapping["Dec_deg"]],
                        name=name,
                        mode="markers",
                        marker=dict(size=2))
                data.append(trace)
        else:
            trace = plotly.graph_objs.Scatter(
                x=dataframe[columnMapping["RA_deg"]],
                y=dataframe[columnMapping["Dec_deg"]],
                mode="markers",
                text=dataframe[columnMapping["name"]],
                marker=dict(size=2)
            )
            data.append(trace)
            
        layout = dict(
            width=550,
            height=550,
            autosize=False,
            title="",
            scene=dict(
                xaxis=dict(
                    title="RA [deg]",
                ),
                yaxis=dict(
                    title="Dec [deg]",
                ),
                aspectratio = dict(x=1, y=1)))
        
        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)
   
    else:
        fig, ax = plt.subplots(1, 1, dpi=200)
        if colorByObject is True:
            a, b = np.unique(dataframe[columnMapping["name"]].values, return_inverse=True)
            hex_map = np.array(sns.color_palette("Accent", len(a)).as_hex())
            c = hex_map[b]
            ax.text(-0.018, 0.016, "Num Objects: {}".format(len(a)), fontsize=8)
        else:
            c = "blue"

        dataframe.plot(x=columnMapping["RA_deg"], y=columnMapping["Dec_deg"], kind="scatter", c=c, s=0.5, ax=ax)
        _setAxes(ax, "equatorialAngular")
    
    
    if usePlotly is True:
        return fig
    else: 
        return fig, ax
        


def plotObservations3D(dataframe, 
                       colorByObject=False, 
                       columnMapping=Config.columnMapping):
    """
    Plot observations in 3D. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    colorByObject : bool, optional
        Color each unique object separately. 
        [Default = False]
    columnMapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~rascals.Config.columnMapping`]
   
    Returns
    -------
    fig : `~plotly.figure`
        Returns the plotly figure object.

    """
    data = []
    if colorByObject is True:
        for name in dataframe[columnMapping["name"]].unique():
            obj = dataframe[dataframe[columnMapping["name"]] == name]

            if name == "NS":
                 trace = plotly.graph_objs.Scatter3d(
                    x=obj[columnMapping["RA_deg"]],
                    y=obj[columnMapping["Dec_deg"]],
                    z=obj[columnMapping["exp_mjd"]] - dataframe[columnMapping["exp_mjd"]].min(),
                    name=name,
                    mode="markers",
                    marker=dict(size=2)
                )
            else:
                trace = plotly.graph_objs.Scatter3d(
                    x=obj[columnMapping["RA_deg"]],
                    y=obj[columnMapping["Dec_deg"]],
                    z=obj[columnMapping["exp_mjd"]] - dataframe[columnMapping["exp_mjd"]].min(),
                    name=name,
                    mode="lines+markers",
                    marker=dict(size=2,
                                line=dict(width=4))
                )
            data.append(trace)
    else:
        trace = plotly.graph_objs.Scatter3d(
            x=dataframe[columnMapping["RA_deg"]],
            y=dataframe[columnMapping["Dec_deg"]],
            z=dataframe[columnMapping["exp_mjd"]] - dataframe[columnMapping["exp_mjd"]].min(),
            mode="markers",
            marker=dict(size=2)
            )
        data.append(trace)

    layout = dict(
        width=800,
        height=550,
        autosize=False,
        title="",
        scene=dict(
            xaxis=dict(
                title="RA [deg]",
            ),
            yaxis=dict(
                title="Dec [deg]",
            ),
            zaxis=dict(
                title="Days [MJD]",
            ),
            aspectratio = dict(x=1, y=1, z=1)))

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return fig
    
