from ..coordinates import Coordinates

__all__ = [
    "Projection"
]

class Projection(Coordinates):

    def __init__(self):
        return

    def to_cartesian(self):
        raise NotImplementedError

    def from_cartesian(cls, cartesian):
        raise NotImplementedError
