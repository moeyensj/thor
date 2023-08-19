import astropy.time
import numpy as np
from typing import TypeAlias
import abc
import quivr as qv
import pyarrow as pa

from adam_core.observations import detections, exposures
from adam_core import orbits, propagator, observers, coordinates
from adam_core.utils.helpers import orbits as util_orbits
from adam_core.coordinates import transform, origin

from . import projections


SourceData: TypeAlias = qv.Linkage[detections.PointSourceDetections, exposures.Exposures]


some_orbits = util_orbits.make_real_orbits(10)
some_observers = observers.Observers.from_code("I11", astropy.time.Time([
    "2020-01-01T00:00:00",
    "2020-01-02T00:00:01",
    "2020-01-03T00:00:02",
    "2020-01-04T00:00:03",
    "2020-01-05T00:00:04",
]))

def make_fake_observations():
    prop = propagator.PYOORB()
    ephems = prop.generate_ephemeris(some_orbits, some_observers).left_table

    exposure_ids = []
    for i in range(len(some_observers)):
        for j in range(len(some_orbits)):
            exposure_id = f"exposure_{i}"
            exposure_ids.append(exposure_id)

    dets = detections.PointSourceDetections.from_kwargs(
        id=[str(i) for i in range(len(ephems))],
        exposure_id=exposure_ids,
        time=ephems.coordinates.time,
        ra=ephems.coordinates.lon,
        ra_sigma=ephems.coordinates.sigma_lon,
        dec=ephems.coordinates.lat,
        dec_sigma=ephems.coordinates.sigma_lat,
        mag=[20 for i in range(len(ephems))],
    )

    unique_exposure_ids = []
    for i in range(len(some_observers)):
        exposure_id = f"exposure_{i}"
        unique_exposure_ids.append(exposure_id)
    exps = exposures.Exposures.from_kwargs(
        id=unique_exposure_ids,
        start_time=some_observers.coordinates.time,
        duration=[30 for i in range(len(some_observers))],
        filter=["i" for i in range(len(some_observers))],
        observatory_code=some_observers.code,
    )

    link = qv.Linkage(
        dets,
        exps,
        left_keys=dets.exposure_id,
        right_keys=exps.id,
    )
    return link

some_observations = make_fake_observations()
    


class ObservationSource(abc.ABC):
    @abc.abstractmethod
    def gather_observations(self, test_orbit: orbits.Orbits) -> SourceData:
        pass

class CellObservationSource(ObservationSource):
    def __init__(self, radius: float):
        """
        radius: The radius of the cell in degrees
        """
        self.radius = radius

    def gather_observations(self, test_orbit: orbits.Orbits) -> SourceData:
        raise NotImplementedError


class StaticObservationsSource(ObservationSource):
    def gather_observations(self, test_orbit: orbits.Orbits) -> SourceData:
        return some_observations

    

def main(src: ObservationSource, test_orbits: orbits.Orbits):
    for o in test_orbits:
        observations = src.gather_observations(o)
        result = range_and_shift(observations, o)
    return result

def range_and_shift(
        observations: SourceData,
        test_orbit: orbits.Orbits,
        prop: propagator.Propagator,
        propagation_chunk_size: int = 100,
        propagation_num_processes: int = 1,
) -> projections.GnomonicCoordinates:
    """
    1. Propagate the test orbit to all epochs in observations
    2. Set the heliocentric distance for each epoch's observations
      to the heliocentric distance of the test orbit at that epoch.
    3. Do the gnomonic dance on the observations
    4, maybe, some linkage? No, because we'll let the caller do that;
    Gnomonic Coordinates contain a observation ID.

    Guarantee that the order of output GnomonicCoordinates is the same as the order of observations.left_table (that is, the original detections).
    """

    # 1. Propagate the test orbit to all epochs in observations. Also
    # generate ephems.
    
    propagation_times = observations.right_table.midpoint().to_astropy()

    propagated_orbit = prop.propagate_orbits(
        test_orbit,
        propagation_times,
        chunk_size=propagation_chunk_size,
        max_processes=propagation_num_processes,
    )

    # TODO: do our own ephem generation off of the state vectors
    # above. For now, we'll use the propagator to generate ephems.

    # First, figure out observers per epoch. This is in the same order
    # as propagation_times, so we don't need extra bookkeeping.
    observers = observations.right_table.observers()

    ephems_linkage: qv.Linkage[orbits.Ephemeris, observers.Observers] = prop.generate_ephemeris(
        test_orbit,
        observers,
        chunk_size=propagation_chunk_size,
        max_processes=propagation_num_processes,
    )

    class EphemTable(qv.Table):
        ephem =  orbits.Ephemeris.as_column()
        observer = observers.Observers.as_column()
        exposure = exposures.Exposures.as_column()
        dets = qv.ListColumn(detections.PointSourceDetections.as_column())

        def from_linkage(linkage: qv.Linkage[orbits.Ephemeris, observers.Observers]) -> "EphemTable":
            return EphemTable(
                ephem=linkage.left_table,
                observer=linkage.right_table,
            )

    ephem_table.dets = [PSD(size=10, frame="ecliptic"), PSD(size=20, frame="equatorial")]
    ephem_table[0].dets -> [PSD(size=10)]
    ephem_table[0].ephem -> Ephemeris(size=1)

    data = qv.Linkage(ephem_table, detections


    # 2. Set the heliocentric distance for each epoch's observations
    # to the heliocentric distance of the test orbit at that epoch.

    # Link the ephems to the exposures. But not all in one go!
    observers = ephems_linkage.right_table
    exposures = observations.right_table
    observers_to_exposures = qv.MultiKeyLinkage(
        observers,
        exposures,
        left_keys={
            "observatory_code": observers.code,
            "time": observers.coordinates.time.jd(),
        },
        right_keys={
            "observatory_code": exposures.observatory_code,
            "time": exposures.midpoint().jd(),
        }
    )

    # Reset time scale to be UTC because we'll use that when
    # associating exposures and propagated orbits
    coords = propagated_orbit.coordinates
    coords.time = coords.time.to_scale("utc")
    propagated_orbit.coordinates = coords

    detections_cartesian = []
    original_ordering = {obs_id.as_py(): i for i, obs_id in enumerate(observations.left_table.id)}
    gnomonic_ordering = []
    for _, ephems, observers in ephems_linkage.iterate():
        assert len(observers) == 1
        assert len(ephems) == 1
        exposure_key = observers_to_exposures.key(
            observatory_code=observers.code[0].as_py(),
            time=ephems.coordinates.time.jd()[0].as_py(),
        )
        exposure = observers_to_exposures.select_right(exposure_key)
        assert len(exposure) == 1
        obs_i = observations.select_left(exposure.id[0].as_py())

        # Now we have a single ephem, a single orbit, and a bunch of
        # observations. Warp the observations based on the orbit state.
        orbit_i = propagated_orbit.select_by_mjd(exposure.midpoint().mjd()[0])

        # Convert RA and Dec to cartesian with an assumed heliocentric
        # distance.

        distance = orbit_i.coordinates.r_mag
        obscodes = pa.repeat(exposure.observatory_code[0], len(obs_i))
        spherical_coords = obs_i.to_spherical(obscodes)

        # HACK: We're plugging in the heliocentric distance to a spot
        # that's taking observer-centric distance. Just because we
        # haven't implemented law of cosines.
        spherical_coords.rho = pa.repeat(distance[0], len(spherical_coords))
        spherical_coords.origin = origin.Origin.from_kwargs(
            code=pa.repeat("SUN", len(spherical_coords)),
        )

        cartesian_detections = transform.transform_coordinates(
            spherical_coords,
            coordinates.CartesianCoordinates,
            frame_out="ecliptic",
            origin_out=origin.OriginCodes.SUN,
        )

        detections_cartesian.append(cartesian_detections)

    detections_cartesian = qv.concatenate(detections_cartesian)

    gnomonic_coords = gnomonic.GnomonicCoordinates.from_cartesian(
        detections_cartesian,
        center_cartesian=propagated_orbit
    )

    return gnomonic_coords
