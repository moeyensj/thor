from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pyarrow.compute as pc

from .analysis import ObservationLabels
from .observations import Observations
from .range_and_transform import TransformedDetections

__all__ = ["plot_transformed_detections"]


COLOR_PALETTE = [
    "blue",
    "green",
    "purple",
    "orange",
    "cyan",
    "magenta",
    "lime",
    "navy",
    "teal",
    "aqua",
    "fuchsia",
    "gold",
    "indigo",
    "turquoise",
    "violet",
    "lavender",
    "chartreuse",
    "steelblue",
    "slateblue",
    "mediumblue",
    "dodgerblue",
    "deepskyblue",
    "lightskyblue",
    "cadetblue",
    "darkturquoise",
    "mediumturquoise",
    "darkseagreen",
    "mediumseagreen",
    "seagreen",
    "forestgreen",
]


def plot_observations(
    observations: Observations,
    labels: Optional[ObservationLabels] = None,
    include_unlabeled: bool = False,
    connect_by_time: bool = True,
    labeled_marker_size: int = 2,
    labeled_line_width: int = 2,
    unlabeled_marker_size: int = 1,
    include_time: bool = True,
) -> go.Figure:
    """
    Plots observations (RA/Dec) and labels.

    Parameters
    ----------
    observations : Observations
        The observations to plot.
    labels : ObservationLabels, optional
        The labels to plot, can be empty if no labels are available.
    include_unlabeled : bool, optional
        Whether to include the unlabeled observations.
    connect_by_time : bool, optional
        Whether to connect the observations by time.
    labeled_marker_size : int, optional
        The size of the labeled markers.
    labeled_line_width : int, optional
        The width of the labeled lines.
    unlabeled_marker_size : int, optional
        The size of the unlabeled markers.
    include_time : bool, optional
        Whether to include the time axis.

    Returns
    -------
    fig : go.Figure
        The figure containing the plotted observations and labels.
    """
    if labels is not None:
        labels = labels.apply_mask(pc.is_in(labels.obs_id, observations.id))

        null_labels = labels.apply_mask(pc.is_null(labels.object_id))
        mask = pc.is_in(observations.id, null_labels.obs_id)

        observations_unlabeled = observations.apply_mask(mask)

        object_ids = labels.apply_mask(pc.invert(pc.is_null(labels.object_id))).object_id.unique()
        observations_labeled = observations.apply_mask(pc.invert(mask))
    else:
        observations_unlabeled = observations
        observations_labeled = observations
        object_ids = []

    fig = go.Figure()

    if len(object_ids) > 0:
        for idx, object_id in enumerate(object_ids.to_pylist()):

            obs_ids = labels.select("object_id", object_id).obs_id
            object_observations = observations_labeled.apply_mask(pc.is_in(observations_labeled.id, obs_ids))
            object_observations = object_observations.sort_by(
                ["coordinates.time.days", "coordinates.time.nanos"]
            )

            o = np.full(len(object_observations), object_id)
            t = object_observations.coordinates.time.mjd().to_numpy(zero_copy_only=False)
            ra = object_observations.coordinates.lon.to_numpy(zero_copy_only=False)
            dec = object_observations.coordinates.lat.to_numpy(zero_copy_only=False)
            i = object_observations.id.to_numpy(zero_copy_only=False)

            customdata = np.stack([o, i, t], axis=1)
            hovertemplate = (
                "object_id=%{customdata[0]}<br>"
                "obs_id=%{customdata[1]}<br>"
                "mjd=%{customdata[2]:.5f}<br>"
                "RA=%{x:.6f}°<br>"
                "Dec=%{y:.6f}°<extra></extra>"
            )

            # Assign color from palette, cycling if needed
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            if include_time:
                fig.add_trace(
                    go.Scatter3d(
                        x=ra,
                        y=dec,
                        z=t,
                        mode="markers+lines" if connect_by_time else "markers",
                        name=str(object_id),
                        marker=dict(size=labeled_marker_size, color=color),
                        line=dict(width=labeled_line_width, color=color),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=ra,
                        y=dec,
                        mode="markers+lines" if connect_by_time else "markers",
                        name=str(object_id),
                        marker=dict(size=labeled_marker_size, color=color),
                        line=dict(width=labeled_line_width, color=color),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )

    if include_unlabeled:

        ra = observations_unlabeled.coordinates.lon.to_numpy(zero_copy_only=False)
        dec = observations_unlabeled.coordinates.lat.to_numpy(zero_copy_only=False)
        t = observations_unlabeled.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        i = observations_unlabeled.id.to_numpy(zero_copy_only=False)

        customdata = np.stack([i, t], axis=1)
        hovertemplate = (
            "obs_id=%{customdata[0]}<br>"
            "mjd=%{customdata[1]:.5f}<br>"
            "RA=%{x:.6f}°<br>"
            "Dec=%{y:.6f}°<extra></extra>"
        )

        if include_time:
            fig.add_trace(
                go.Scatter3d(
                    x=ra,
                    y=dec,
                    z=t,
                    mode="markers",
                    marker=dict(size=unlabeled_marker_size, color="lightcoral"),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Unknown",
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=ra,
                    y=dec,
                    mode="markers",
                    marker=dict(size=unlabeled_marker_size, color="lightcoral"),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Unknown",
                )
            )

    title = "Observations (RA, Dec)"

    if include_time:
        fig.update_layout(
            title=title,
            paper_bgcolor="black",
            width=1000,
            height=1000,
            title_font=dict(color="white"),
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
            scene=dict(
                bgcolor="black",
                xaxis=dict(
                    title=dict(text="RA [deg]"),
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
                yaxis=dict(
                    title=dict(text="Dec [deg]"),
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
                zaxis=dict(
                    title="Time [MJD]",
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
            ),
        )
    else:
        fig.update_layout(
            title=title,
            paper_bgcolor="black",
            plot_bgcolor="black",
            width=1000,
            height=1000,
            title_font=dict(color="white"),
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(
                title=dict(text="RA [deg]"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                zeroline=False,
                linecolor="white",
                color="white",
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                mirror=True,
            ),
            yaxis=dict(
                title=dict(text="Dec [deg]"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                zeroline=False,
                linecolor="white",
                color="white",
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                mirror=True,
                scaleanchor="x",
                scaleratio=1,
            ),
        )
    return fig


def plot_transformed_detections(
    transformed_detections: TransformedDetections,
    labels: Optional[ObservationLabels] = None,
    include_unlabeled: bool = False,
    connect_by_time: bool = True,
    labeled_marker_size: int = 2,
    labeled_line_width: int = 2,
    unlabeled_marker_size: int = 1,
    include_time: bool = True,
) -> go.Figure:
    """
    Plots the transformed detections and labels.

    Parameters
    ----------
    transformed_detections : TransformedDetections
        The transformed detections to plot.
    labels : ObservationLabels, optional
        The labels to plot, can be empty if no labels are available.
    include_unlabeled : bool, optional
        Whether to include the unlabeled detections.
    connect_by_time : bool, optional
        Whether to connect the detections by time.
    labeled_marker_size : int, optional
        The size of the labeled markers.
    labeled_line_width : int, optional
        The width of the labeled lines.
    unlabeled_marker_size : int, optional
        The size of the unlabeled markers.
    include_time : bool, optional
        Whether to include the time axis.

    Returns
    -------
    fig : go.Figure
        The figure containing the plotted detections and labels.
    """
    if labels is not None:
        labels = labels.apply_mask(pc.is_in(labels.obs_id, transformed_detections.id))

        null_labels = labels.apply_mask(pc.is_null(labels.object_id))
        mask = pc.is_in(transformed_detections.id, null_labels.obs_id)

        transformed_detections_unlabeled = transformed_detections.apply_mask(mask)

        object_ids = labels.apply_mask(pc.invert(pc.is_null(labels.object_id))).object_id.unique()
        transformed_detections_labeled = transformed_detections.apply_mask(pc.invert(mask))
    else:
        transformed_detections_unlabeled = transformed_detections
        transformed_detections_labeled = transformed_detections
        object_ids = []

    fig = go.Figure()

    if len(object_ids) > 0:
        for idx, object_id in enumerate(object_ids.to_pylist()):

            obs_ids = labels.select("object_id", object_id).obs_id
            object_detections = transformed_detections_labeled.apply_mask(
                pc.is_in(transformed_detections_labeled.id, obs_ids)
            )
            object_detections = object_detections.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

            o = np.full(len(object_detections), object_id)
            t = object_detections.coordinates.time.mjd().to_numpy(zero_copy_only=False)
            x = object_detections.coordinates.theta_x.to_numpy(zero_copy_only=False)
            y = object_detections.coordinates.theta_y.to_numpy(zero_copy_only=False)
            n = object_detections.night.to_numpy(zero_copy_only=False)
            i = object_detections.id.to_numpy(zero_copy_only=False)

            customdata = np.stack([o, i, t, n], axis=1)
            hovertemplate = (
                "object_id=%{customdata[0]}<br>"
                "obs_id=%{customdata[1]}<br>"
                "mjd=%{customdata[2]:.5f}<br>"
                "night=%{customdata[3]}<br>"
                "θx=%{x:.6f}°<br>"
                "θy=%{y:.6f}°<extra></extra>"
            )

            # Assign color from palette, cycling if needed
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

            if include_time:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=t,
                        mode="markers+lines" if connect_by_time else "markers",
                        name=str(object_id),
                        marker=dict(size=labeled_marker_size, color=color),
                        line=dict(width=labeled_line_width, color=color),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )
            else:
                fig.add_trace(
                    go.Scattergl(
                        x=x,
                        y=y,
                        mode="markers+lines" if connect_by_time else "markers",
                        name=str(object_id),
                        marker=dict(size=labeled_marker_size, color=color),
                        line=dict(width=labeled_line_width, color=color),
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    )
                )

    if include_unlabeled:

        x = transformed_detections_unlabeled.coordinates.theta_x.to_numpy(zero_copy_only=False)
        y = transformed_detections_unlabeled.coordinates.theta_y.to_numpy(zero_copy_only=False)
        t = transformed_detections_unlabeled.coordinates.time.mjd().to_numpy(zero_copy_only=False)
        n = transformed_detections_unlabeled.night.to_numpy(zero_copy_only=False)
        i = transformed_detections_unlabeled.id.to_numpy(zero_copy_only=False)

        customdata = np.stack([i, t, n], axis=1)
        hovertemplate = (
            "obs_id=%{customdata[0]}<br>"
            "mjd=%{customdata[1]:.5f}<br>"
            "night=%{customdata[2]}<br>"
            "θx=%{x:.6f}°<br>"
            "θy=%{y:.6f}°<extra></extra>"
        )

        if include_time:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=t,
                    mode="markers",
                    marker=dict(size=unlabeled_marker_size, color="lightcoral"),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Unknown",
                )
            )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=unlabeled_marker_size, color="lightcoral"),
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name="Unknown",
                )
            )

    title = r"Transformed Detections (θ<sub>X</sub>, θ<sub>Y</sub>)"

    if include_time:
        fig.update_layout(
            title=title,
            paper_bgcolor="black",
            width=1000,
            height=1000,
            title_font=dict(color="white"),
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
            scene=dict(
                bgcolor="black",
                xaxis=dict(
                    title=dict(text="θ<sub>X</sub> [deg]"),
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
                yaxis=dict(
                    title=dict(text="θ<sub>Y</sub> [deg]"),
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
                zaxis=dict(
                    title="Time [MJD]",
                    showbackground=False,
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.10)",
                    zerolinecolor="white",
                    linecolor="white",
                    color="white",
                    title_font=dict(color="white"),
                ),
            ),
        )
    else:
        fig.update_layout(
            title=title,
            paper_bgcolor="black",
            plot_bgcolor="black",
            width=1000,
            height=1000,
            title_font=dict(color="white"),
            legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(
                title=dict(text="θ<sub>X</sub> [deg]"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                zeroline=False,
                linecolor="white",
                color="white",
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                mirror=True,
            ),
            yaxis=dict(
                title=dict(text="θ<sub>Y</sub> [deg]"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                zeroline=False,
                linecolor="white",
                color="white",
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                mirror=True,
                scaleanchor="x",
                scaleratio=1,
            ),
        )
    return fig
