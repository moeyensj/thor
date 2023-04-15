import os

import numpy as np
import pandas as pd
from astropy.time import Time

from ...data_processing import preprocessObservations
from ..ephemeris import generateEphemeris
from ..iod import initialOrbitDetermination
from ..orbits import Orbits

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../testing/data"
)
