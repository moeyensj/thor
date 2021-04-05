import plotly
import numpy as np

from ..orbits import propagateOrbits
from ..orbits import getPerturberState

__all__ = [
    "plotOrbits"
]

PLANET_COLORS = {
    "mercury" : "#7F4D21",
    "venus" : "#E7B765",
    "earth" : "#5B6C8B",
    "mars barycenter" : "#D84325",
    "jupiter barycenter" : "#DDB282",
    "saturn barycenter" : "#E3C299",
    "uranus barycenter" : "#82B7CE",
    "neptune barycenter" : "#5A63F4",
}
DTS = np.arange(-60, 1, 5)

def addPerturber(perturber, t0, dts, color=None):
    
    # Set perturber's name
    if perturber.find("barycenter"):
        name = perturber.split(" ")[0]
    name = name.capitalize()
    
    if not isinstance(color, str):
        color = "white"
    
    perturber_data = []
    if perturber == "sun":
        trace = plotly.graph_objs.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            name=name,
            mode="markers",
            marker=dict(
                size=4,
                color=color
            ),
        )
        perturber_data.append(trace)
        
    else:
        perturber_state_t0 = getPerturberState(perturber, t0)
        trace = plotly.graph_objs.Scatter3d(
            x=perturber_state_t0[:, 0],
            y=perturber_state_t0[:, 1],
            z=perturber_state_t0[:, 2],
            name=name,
            mode="markers",
            marker=dict(
                size=3,
                color=color
            ),
            hovertemplate =
                '%{text}<br>'+ 
                'x: %{x}<br>'+
                'y: %{y}<br>'+
                'z: %{z}<br>',
            text = ["MJD [TDB]: {:.5f}".format(i) for i in t0.tdb.mjd],
        )
        perturber_data.append(trace)
        
        
        t1 = t0 + dts
        perturber_states = getPerturberState(perturber, t1)
        trace = plotly.graph_objs.Scatter3d(
            x=perturber_states[:, 0],
            y=perturber_states[:, 1],
            z=perturber_states[:, 2],
            name=name,
            mode="markers",
            marker=dict(
                size=1,
                color=color
            ),
            hovertemplate =
                '%{text}<br>'+ 
                'x: %{x}<br>'+
                'y: %{y}<br>'+
                'z: %{z}<br>',
            text = ["MJD [TDB]: {:.5f}".format(i) for i in t1.tdb.mjd],
        )
        perturber_data.append(trace)

    return perturber_data


def addOrbits(orbits, dts):
    
    assert len(np.unique(orbits.epochs)) == 1
    
    orbit_data = []
    t1 = orbits.epochs[0] + dts
    propagated = propagateOrbits(orbits, t1)
    
    for orbit_id in orbits.ids:
    
        trace = plotly.graph_objs.Scatter3d(
            x=orbits.cartesian[np.where(orbits.ids == orbit_id)[0], 0],
            y=orbits.cartesian[np.where(orbits.ids == orbit_id)[0], 1],
            z=orbits.cartesian[np.where(orbits.ids == orbit_id)[0], 2],
            name=orbit_id,
            mode="markers",
            marker=dict(
                size=2,
                color="white"
            ),
            hovertemplate =
                '%{text}<br>'+ 
                'x: %{x}<br>'+
                'y: %{y}<br>'+
                'z: %{z}<br>',
            text = ["MJD [TDB]: {:.5f}".format(i) for i in t1.tdb.mjd],
        )
        orbit_data.append(trace)
        
        propagated_mask =( propagated["orbit_id"] == orbit_id)
        trace = plotly.graph_objs.Scatter3d(
            x=propagated[propagated_mask]["x"].values,
            y=propagated[propagated_mask]["y"].values,
            z=propagated[propagated_mask]["z"].values,
            name=orbit_id,
            mode="markers",
            marker=dict(
                size=1,
                color="white"
            ),
            hovertemplate =
                '%{text}<br>'+ 
                'x: %{x}<br>'+
                'y: %{y}<br>'+
                'z: %{z}<br>',
            text = ["MJD [TDB]: {:.5f}".format(i) for i in t1.tdb.mjd],
        )
        orbit_data.append(trace)
    
    return orbit_data

def plotOrbits(
        orbits, 
        dts=DTS, 
        inner_planets=True, 
        outer_planets=True
    ):
    
    gridcolor = "rgb(96,96,96)"
    zerolinecolor = gridcolor

    data = []
    data += addOrbits(orbits, dts)
    limits = (-5, 5)
    
    t0 = orbits.epochs[:1]
    data += addPerturber("sun", t0, dts, color="#FFD581")
    
    if inner_planets:
        for perturber in ["mercury", "venus", "earth", "mars barycenter"]:
            data += addPerturber(perturber, t0, dts, color=PLANET_COLORS[perturber])
                        
    if outer_planets:
        for perturber in ["jupiter barycenter", "saturn barycenter", "uranus barycenter", "neptune barycenter"]:
            data += addPerturber(perturber, t0, dts, color=PLANET_COLORS[perturber])
            
        limits = (-50, 50)

    layout = dict(
        width=1000,
        height=1000,
        autosize=False,
        title="",
        scene=dict(
            xaxis=dict(
                title="x [au]",
                gridcolor=gridcolor,
                zerolinecolor=zerolinecolor,
                showbackground=False,
                range=limits
            ),
            yaxis=dict(
                title="y [au]",
                gridcolor=gridcolor,
                zerolinecolor=zerolinecolor,
                showbackground=False,
                range=limits
            ),
            zaxis=dict(
                title="z [au]",
                gridcolor=gridcolor,
                zerolinecolor=zerolinecolor,
                showbackground=False,
                range=limits
            ),
            aspectratio=dict(
                x=1, 
                y=1, 
                z=1
            ),

        ),
        font_color="white",
        plot_bgcolor="rgb(0,0,0)",
        paper_bgcolor="rgba(0,0,0)",
        showlegend=False

    )

    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return fig