import plotly
import pandas as pd
import matplotlib.pyplot as plt

from ..config import Config
from .helpers import _setAxes

__all__ = [
    "plotObservations",
    "plotObservations3D"
]

def plotObservations(
        dataframe, 
        color_by_object=False, 
        use_plotly=True, 
        column_mapping=Config.COLUMN_MAPPING
    ):
    """
    Plot observations in 2D. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    color_by_object : bool, optional
        Color each unique object separately. 
        [Default = False]
    use_plotly : bool, optional
        Use plotly instead of matplotlib?
        [Default = True]
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.COLUMN_MAPPING`]
   
    Returns
    -------
    fig : {`~matplotlib.figure.Figure`, `~plotly.figure`}
        Returns the matplotlib or plotly figure object depending
        on the selected plotting package.
    ax : `~matplotlib.axes._subplots.AxesSubplot`
        If plotting with matplotlib, also returns the axis object.
    """
    if use_plotly is True:
        data = []
        if color_by_object is True:
            for name in dataframe[column_mapping["name"]].unique():
                obj = dataframe[dataframe[column_mapping["name"]] == name]
                if name == "NS":
                    trace = plotly.graph_objs.Scatter(
                        x=obj[column_mapping["RA_deg"]],
                        y=obj[column_mapping["Dec_deg"]],
                        name=name,
                        mode="markers",
                        marker=dict(size=2))
                else:
                    trace = plotly.graph_objs.Scatter(
                        x=obj[column_mapping["RA_deg"]],
                        y=obj[column_mapping["Dec_deg"]],
                        name=name,
                        mode="markers",
                        marker=dict(size=2))
                data.append(trace)
        else:
            trace = plotly.graph_objs.Scatter(
                x=dataframe[column_mapping["RA_deg"]],
                y=dataframe[column_mapping["Dec_deg"]],
                mode="markers",
                text=dataframe[column_mapping["name"]],
                marker=dict(size=2)
            )
            data.append(trace)
            
        layout = dict(
            width=1000,
            height=1000,
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
        fig, ax = plt.subplots(1, 1, dpi=600)
        if color_by_object is True:
            a, b = np.unique(dataframe[column_mapping["name"]].values, return_inverse=True)
            hex_map = np.array(sns.color_palette("Accent", len(a)).as_hex())
            c = hex_map[b]
            ax.text(-0.018, 0.016, "Num Objects: {}".format(len(a)), fontsize=8)
        else:
            c = "blue"

        dataframe.plot(x=column_mapping["RA_deg"], y=column_mapping["Dec_deg"], kind="scatter", c=c, s=0.5, ax=ax)
        _setAxes(ax, "equatorialAngular")
    
    
    if use_plotly is True:
        return fig
    else: 
        return fig, ax
        


def plotObservations3D(
        dataframe, 
        color_by_object=False, 
        column_mapping=Config.COLUMN_MAPPING
    ):
    """
    Plot observations in 3D. 
    
    Parameters
    ----------
    dataframe : `~pandas.DataFrame`
        DataFrame containing relevant quantities to be plotted.
    color_by_object : bool, optional
        Color each unique object separately. 
        [Default = False]
    column_mapping : dict, optional
        Column name mapping of observations to internally used column names. 
        [Default = `~thor.Config.COLUMN_MAPPING`]
   
    Returns
    -------
    fig : `~plotly.figure`
        Returns the plotly figure object.

    """
    data = []
    if color_by_object is True:
        for name in dataframe[column_mapping["name"]].unique():
            obj = dataframe[dataframe[column_mapping["name"]] == name]

            if name == "NS":
                 trace = plotly.graph_objs.Scatter3d(
                    x=obj[column_mapping["RA_deg"]],
                    y=obj[column_mapping["Dec_deg"]],
                    z=obj[column_mapping["exp_mjd"]] - dataframe[column_mapping["exp_mjd"]].min(),
                    name=name,
                    mode="markers",
                    marker=dict(size=2)
                )
            else:
                trace = plotly.graph_objs.Scatter3d(
                    x=obj[column_mapping["RA_deg"]],
                    y=obj[column_mapping["Dec_deg"]],
                    z=obj[column_mapping["exp_mjd"]] - dataframe[column_mapping["exp_mjd"]].min(),
                    name=name,
                    mode="lines+markers",
                    marker=dict(size=2,
                                line=dict(width=4))
                )
            data.append(trace)
    else:
        trace = plotly.graph_objs.Scatter3d(
            x=dataframe[column_mapping["RA_deg"]],
            y=dataframe[column_mapping["Dec_deg"]],
            z=dataframe[column_mapping["exp_mjd"]] - dataframe[column_mapping["exp_mjd"]].min(),
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
    
