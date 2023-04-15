import matplotlib.pyplot as plt
import pandas as pd
import plotly

from ..config import Config
from .helpers import _setAxes

__all__ = ["plotObservations", "plotObservations3D"]


def plotObservations(
    dataframe,
    color_by_object=False,
    use_plotly=True,
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
                        x=obj["RA_deg"],
                        y=obj["Dec_deg"],
                        name=name,
                        mode="markers",
                        marker=dict(size=2),
                    )
                else:
                    trace = plotly.graph_objs.Scatter(
                        x=obj["RA_deg"],
                        y=obj["Dec_deg"],
                        name=name,
                        mode="markers",
                        marker=dict(size=2),
                    )
                data.append(trace)
        else:
            trace = plotly.graph_objs.Scatter(
                x=dataframe["RA_deg"],
                y=dataframe["Dec_deg"],
                mode="markers",
                text=dataframe[column_mapping["name"]],
                marker=dict(size=2),
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
                aspectratio=dict(x=1, y=1),
            ),
        )

        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    else:
        fig, ax = plt.subplots(1, 1, dpi=600)
        if color_by_object is True:
            a, b = np.unique(
                dataframe[column_mapping["name"]].values, return_inverse=True
            )
            hex_map = np.array(sns.color_palette("Accent", len(a)).as_hex())
            c = hex_map[b]
            ax.text(-0.018, 0.016, "Num Objects: {}".format(len(a)), fontsize=8)
        else:
            c = "blue"

        dataframe.plot(x="RA_deg", y="Dec_deg", kind="scatter", c=c, s=0.5, ax=ax)
        _setAxes(ax, "equatorialAngular")

    if use_plotly is True:
        return fig
    else:
        return fig, ax


def plotObservations3D(
    dataframe,
    color_by_object=False,
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
                    x=obj["RA_deg"],
                    y=obj["Dec_deg"],
                    z=obj["mjd_utc"] - dataframe["mjd_utc"].min(),
                    name=name,
                    mode="markers",
                    marker=dict(size=2),
                )
            else:
                trace = plotly.graph_objs.Scatter3d(
                    x=obj["RA_deg"],
                    y=obj["Dec_deg"],
                    z=obj["mjd_utc"] - dataframe["mjd_utc"].min(),
                    name=name,
                    mode="lines+markers",
                    marker=dict(size=2, line=dict(width=4)),
                )
            data.append(trace)
    else:
        trace = plotly.graph_objs.Scatter3d(
            x=dataframe["RA_deg"],
            y=dataframe["Dec_deg"],
            z=dataframe["mjd_utc"] - dataframe["mjd_utc"].min(),
            mode="markers",
            marker=dict(size=2),
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
            aspectratio=dict(x=1, y=1, z=1),
        ),
    )

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return fig
