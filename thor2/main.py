import dataclasses
import logging
from typing import Protocol, TypeAlias

import quivr as qv
from adam_core import observers, orbits, origin, propagator
from adam_core.observations import detections, exposures

from . import mathutil, rotations

SourceData: TypeAlias = qv.Linkage[
    detections.PointSourceDetections, exposures.Exposures
]


@dataclasses.dataclass
class Configuration:
    # cell_radius is the maximum distance for between an observation
    # and a test orbit to consider the observation worth
    # including. This distance is in degrees, and is based on the sky
    # positions, as observed from earth (at the actual observer
    # location where the detection occurred).
    cell_radius: float


def range_and_shift(
    observations: SourceData,
    test_orbit: Orbits,
    propagator: propagator.Propagator,
    cfg: Configuration,
):
    exposures = observations.right_table

    # Propagate test orbit to all observer times/locations, and
    # generate an ephemeris for each
    test_orbit_ephem_per_observer = propagator.generate_ephemeris(test_orbit, exposures.observers())

    # Link the generated ephemerides back up to the source exposures
    # based on observatory code and timestamp.
    test_orbit_ephems = test_orbit_ephem_per_observer.left_table
    ephem_per_exposure = qv.MultiKeyLinkage(
        left_table=exposures,
        right_table=test_orbit_ephems,
        left_keys={
            "code": exposures.observatory_code,
            "jd1": exposures.midpoint().jd1,
            "jd2": exposures.midpoint().jd2,
        },
        right_keys={
            "code": test_orbit_ephems.coordinates.origin.code,
            "jd1": test_orbit_ephems.coordinates.time.jd1,
            "jd2": test_orbit_ephems.coordinates.time.jd2,
        },
    )

    # Partition off the work to be done
    for obskey, exposures_i, ephems_i in ephem_per_exposure:
        assert len(exposures_i) == 1, "the exposure-ephem mapping should be 1-to-1"
        assert len(ephems_i) == 1, "the exposure-ephem mapping should be 1-to-1"

        exposure_id = exposures_i.id[0]

        observations_i = observations.select(exposure_id)

        ephem_ra = ephems_i.coordinates.lon[0].as_py()
        ephem_dec = ephems_i.coordinates.lat[0].as_py()

        distances = mathutil.angular_separation(
            observations_i.ra, observations_i.dec, ephem_ra, ephem_dec
        )
        observations_i = observations_i.apply_mask(distances < cfg.cell_radius)

        if len(observations_i) > 0:
            # FIXME: We're redoing propagation here; we already did it
            # for ephemeris generation. It should be possible to avoid
            # that repeated work.
            epoch = exposures_i.midpoint()
            test_orbit_state = propagator.propagate_orbits(
                test_orbit, epoch.to_astropy()
            )
            # TODO: could we vectorize this whole thing?
            rotmat = rotations.rotation_matrix(
                test_orbit_state.coordinates.values, epoch
            )

            as_gnomonic = rotmat.apply(observations_i)

            # yield? idk
            yield as_gnomonic
