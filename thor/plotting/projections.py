import matplotlib.pyplot as plt
import pandas as pd
import plotly

from .helpers import _setAxes

__all__ = ["plotProjections", "plotProjections3D"]


def plotProjections(
    projected_observations,
    color_by_object=False,
    use_plotly=True,
):
    """
    Plot projected observations in 2D.

    Parameters
    ----------
    projected_observations :  `~pandas.DataFrame`
        DataFrame containing projected observations (theta_x, theta_y).
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
            for name in projected_observations["obj_id"].unique():
                obj = projected_observations[projected_observations["obj_id"] == name]
                if name == "None":
                    trace = plotly.graph_objs.Scatter(
                        x=obj["theta_x_deg"].values,
                        y=obj["theta_y_deg"].values,
                        name="Unknown",
                        mode="markers",
                        marker=dict(size=2),
                    )
                else:
                    trace = plotly.graph_objs.Scatter(
                        x=obj["theta_x_deg"].values,
                        y=obj["theta_y_deg"].values,
                        name=name,
                        mode="lines+markers",
                        marker=dict(size=2, line=dict(width=2)),
                    )
                data.append(trace)
        else:
            trace = plotly.graph_objs.Scatter(
                x=projected_observations["theta_x_deg"].values,
                y=projected_observations["theta_y_deg"].values,
                mode="markers",
                text=projected_observations["obj_id"].values,
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
                    title="Theta X [deg]",
                ),
                yaxis=dict(
                    title="Theta Y [deg]",
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
                projected_observations["obj_id"].values, return_inverse=True
            )
            hex_map = np.array(sns.color_palette("Accent", len(a)).as_hex())
            c = hex_map[b]
            ax.text(-0.018, 0.016, "Num Objects: {}".format(len(a)), fontsize=8)
        else:
            c = "blue"

        dataframe.plot(
            x="theta_x_deg", y="theta_y_deg", kind="scatter", c=c, s=0.5, ax=ax
        )
        _setAxes(ax, "gnomonic")

    if use_plotly is True:
        return fig
    else:
        return fig, ax


def plotProjections3D(
    projected_observations,
    color_by_object=False,
):
    """
    Plot projected observations in 3D.

    Parameters
    ----------
    projected_observations :  `~pandas.DataFrame`
        DataFrame containing projected observations (theta_x, theta_y).
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
        for name in projected_observations["obj_id"].unique():
            obj = projected_observations[projected_observations["obj_id"] == name]

            if name == "None":
                trace = plotly.graph_objs.Scatter3d(
                    x=obj["theta_x_deg"].values,
                    y=obj["theta_y_deg"].values,
                    z=obj["mjd_utc"].values - projected_observations["mjd_utc"].min(),
                    name="Unknown",
                    mode="markers",
                    marker=dict(size=2),
                )
            else:
                trace = plotly.graph_objs.Scatter3d(
                    x=obj["theta_x_deg"].values,
                    y=obj["theta_y_deg"].values,
                    z=obj["mjd_utc"].values - projected_observations["mjd_utc"].min(),
                    name=name,
                    mode="lines+markers",
                    marker=dict(size=2, line=dict(width=4)),
                )
            data.append(trace)
    else:
        trace = plotly.graph_objs.Scatter3d(
            x=projected_observations["theta_x_deg"].values,
            y=projected_observations["theta_y_deg"].values,
            z=projected_observations["mjd_utc"].values
            - projected_observations["mjd_utc"].min(),
            mode="markers",
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
                title="Theta X [deg]",
            ),
            yaxis=dict(
                title="Theta Y [deg]",
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
