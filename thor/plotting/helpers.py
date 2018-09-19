import matplotlib.pyplot as plt

__all__ = ["_setAxes",
           "_setPercentage"]

def _setAxes(ax, coordinateSystem):
    """
    Helper function to set axes limits depending on the coordinate system.
    
    """
    if coordinateSystem == "equatorialAngular":
        ax.set_xlabel(r"$\alpha$ [deg]")
        ax.set_ylabel(r"$\delta$ [deg]")
    elif coordinateSystem == "eclipticAngular":
        ax.set_xlabel(r"$\lambda$ [deg]")
        ax.set_ylabel(r"$\beta$ [deg]")
    elif coordinateSystem == "gnomonic":
        ax.set_xlabel(r"$\theta_X$ [deg]")
        ax.set_ylabel(r"$\theta_Y$ [deg]")
    else:
        raise ValueError("coordinateSystem should be one of: 'equatorialAngular', 'eclipticAngular', 'tangentPlane'")
    ax.set_aspect("equal")
    return

def _setPercentage(limits, percentage):
    """
    Helper function which returns the percetange value measured from the lower limit.
    For example, _setPercentage([2, 4], 1.0) would return 4 whereas _setPercentage([2, 4], 0.50)
    would return 3.

    This is particularly useful for putting text on a plot. 

    """
    return (limits[-1] - limits[0]) * percentage + limits[0]

