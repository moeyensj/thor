import pandas as pd
from adam_core.orbits import Ephemeris


def _ephemeris_to_dataframe(ephemeris: Ephemeris) -> pd.DataFrame:
    ephemeris = ephemeris.sort_by(
        [
            "orbit_id",
            "coordinates.time.days",
            "coordinates.time.nanos",
            "coordinates.origin.code",
        ]
    )
    ephemeris_df = ephemeris.to_dataframe()
    ephemeris_df.insert(2, "mjd_utc", ephemeris.coordinates.time.mjd())
    ephemeris_df.rename(
        columns={
            "coordinates.lon": "RA_deg",
            "coordinates.lat": "Dec_deg",
            "coordinates.origin.code": "observatory_code",
        },
        inplace=True,
    )
    return ephemeris_df
