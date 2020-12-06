from ..utils import _checkTime

__all__ = ["Orbits"]


class Orbits:

    def __init__(
            self, 
            orbits, 
            epochs,
            orbit_type="cartesian",
            covariances=None,
            ids=None,
            H=None,
            G=None,
        ):
        # Make sure that the given epoch(s) are an astropy time object
        _checkTime(epochs, "epoch")

        # Make sure that each orbit has a an epoch
        assert len(epochs) == orbits.shape[0]
        self.epochs = epochs
        self.num_orbits = orbits.shape[0]

        # Make sure the passed orbits are one of the the supported types
        if orbit_type == "cartesian":
            self.cartesian = orbits
        elif orbit_type == "keplerian":
            self.keplerian = orbits
            self.cartesian = convertOrbitalElements(
                orbits, 
                "keplerian", 
                "cartesian"
            )
        else:
            err = (
                "orbit_type has to be one of {'cartesian', 'keplerian'}."
            )
            raise ValueError(err)
        self.orbit_type = "cartesian"

        # If object IDs have been passed make sure each orbit has one
        if ids:
            assert len(ids) == self.num_orbits
        self.ids = ids

        # If H magnitudes have been passed make sure each orbit has one
        if H:
            assert len(H) == self.num_orbits
        self.H = H
        
        # If the slope parameter G has been passed make sure each orbit has one
        if G: 
            assert len(G) == self.num_orbits
        self.G = G

        # If covariances matrixes have been passed make sure each orbit has one
        if covariances:
            assert len(covariances) == self.num_orbits
        self.covariances = covariances

        return

    def __repr__(self):
        rep = (
            "Orbits: {}\n"
            "Type: {}\n"
        )
        return rep.format(self.num_orbits, self.orbit_type)
