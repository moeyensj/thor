"""
Phase space coverage analysis for THOR test orbits.

This module provides a simple interface for generating test orbits with
even phase space coverage and analyzing their overlap characteristics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.time import Timestamp

from ..orbit import TestOrbits

logger = logging.getLogger(__name__)


@dataclass
class PhaseSpaceBounds:
    """
    Defines bounds for a 6D phase space region in any coordinate system.

    This class can represent bounds in spherical, cartesian, or keplerian coordinates.
    The coordinate system is determined by which parameters are provided.

    Parameters
    ----------
    coordinate_system : str
        The coordinate system: "spherical", "cartesian", or "keplerian"
    bounds : Dict[str, Tuple[float, float]]
        Dictionary mapping coordinate names to (min, max) bounds
    """

    coordinate_system: str
    bounds: Dict[str, Tuple[float, float]]

    def __post_init__(self):
        """Validate the coordinate system and bounds."""
        valid_systems = {
            "spherical": ["rho", "lon", "lat", "vrho", "vlon", "vlat"],
            "cartesian": ["x", "y", "z", "vx", "vy", "vz"],
            "keplerian": ["a", "e", "i", "raan", "ap", "M"],
        }

        if self.coordinate_system not in valid_systems:
            raise ValueError(f"coordinate_system must be one of {list(valid_systems.keys())}")

        expected_coords = valid_systems[self.coordinate_system]
        if set(self.bounds.keys()) != set(expected_coords):
            raise ValueError(
                f"For {self.coordinate_system} coordinates, bounds must contain: {expected_coords}"
            )

    @classmethod
    def from_spherical(
        cls,
        rho: Tuple[float, float],
        lon: Tuple[float, float],
        lat: Tuple[float, float],
        vrho: Tuple[float, float],
        vlon: Tuple[float, float],
        vlat: Tuple[float, float],
    ) -> "PhaseSpaceBounds":
        """Create bounds from spherical coordinate parameters."""
        return cls(
            coordinate_system="spherical",
            bounds={"rho": rho, "lon": lon, "lat": lat, "vrho": vrho, "vlon": vlon, "vlat": vlat},
        )

    @classmethod
    def from_cartesian(
        cls,
        x: Tuple[float, float],
        y: Tuple[float, float],
        z: Tuple[float, float],
        vx: Tuple[float, float],
        vy: Tuple[float, float],
        vz: Tuple[float, float],
    ) -> "PhaseSpaceBounds":
        """Create bounds from cartesian coordinate parameters."""
        return cls(
            coordinate_system="cartesian", bounds={"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}
        )

    @classmethod
    def from_keplerian(
        cls,
        a: Tuple[float, float],
        e: Tuple[float, float],
        i: Tuple[float, float],
        raan: Tuple[float, float],
        ap: Tuple[float, float],
        M: Tuple[float, float],
    ) -> "PhaseSpaceBounds":
        """Create bounds from keplerian coordinate parameters."""
        return cls(
            coordinate_system="keplerian", bounds={"a": a, "e": e, "i": i, "raan": raan, "ap": ap, "M": M}
        )

    @classmethod
    def from_type(cls, asteroid_type: str, coordinate_system: str = "spherical") -> "PhaseSpaceBounds":
        """
        Create bounds for different asteroid populations.

        Parameters
        ----------
        asteroid_type : str
            Type of asteroid population: "main_belt", "near_earth", "jupiter_trojans",
            "comprehensive", or "inner_main_belt"
        coordinate_system : str, optional
            Coordinate system to use. Default: "spherical"

        Returns
        -------
        PhaseSpaceBounds
            Bounds appropriate for the specified asteroid type and coordinate system
        """
        if asteroid_type == "main_belt":
            return cls._main_belt_bounds(coordinate_system)
        elif asteroid_type == "near_earth":
            return cls._near_earth_bounds(coordinate_system)
        elif asteroid_type == "jupiter_trojans":
            return cls._jupiter_trojans_bounds(coordinate_system)
        elif asteroid_type == "comprehensive":
            return cls._comprehensive_bounds(coordinate_system)
        elif asteroid_type == "inner_main_belt":
            return cls._inner_main_belt_bounds(coordinate_system)
        else:
            raise ValueError(
                f"Unknown asteroid type: {asteroid_type}. "
                f"Available types: main_belt, near_earth, jupiter_trojans, comprehensive, inner_main_belt"
            )

    @classmethod
    def _main_belt_bounds(cls, coordinate_system: str) -> "PhaseSpaceBounds":
        """Main belt asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(1.8, 4.0),
                lon=(0.0, 360.0),
                lat=(-30.0, 30.0),
                vrho=(-0.01, 0.01),
                vlon=(-1.0, 1.0),
                vlat=(-1.0, 1.0),
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-4.0, 4.0),
                y=(-4.0, 4.0),
                z=(-1.0, 1.0),
                vx=(-0.02, 0.02),
                vy=(-0.02, 0.02),
                vz=(-0.01, 0.01),
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(1.8, 4.0), e=(0.0, 0.3), i=(0.0, 30.0), raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    @classmethod
    def _near_earth_bounds(cls, coordinate_system: str) -> "PhaseSpaceBounds":
        """Near-Earth asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(0.5, 3.0),
                lon=(0.0, 360.0),
                lat=(-45.0, 45.0),
                vrho=(-0.02, 0.02),
                vlon=(-2.0, 2.0),
                vlat=(-2.0, 2.0),
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-3.0, 3.0),
                y=(-3.0, 3.0),
                z=(-1.5, 1.5),
                vx=(-0.03, 0.03),
                vy=(-0.03, 0.03),
                vz=(-0.02, 0.02),
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(0.5, 3.0), e=(0.0, 0.8), i=(0.0, 45.0), raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    @classmethod
    def _jupiter_trojans_bounds(cls, coordinate_system: str) -> "PhaseSpaceBounds":
        """Jupiter Trojan asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(4.5, 5.8),
                lon=(0.0, 360.0),
                lat=(-25.0, 25.0),
                vrho=(-0.005, 0.005),
                vlon=(-0.5, 0.5),
                vlat=(-0.5, 0.5),
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-6.0, 6.0),
                y=(-6.0, 6.0),
                z=(-2.0, 2.0),
                vx=(-0.015, 0.015),
                vy=(-0.015, 0.015),
                vz=(-0.01, 0.01),
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(4.5, 5.8), e=(0.0, 0.2), i=(0.0, 25.0), raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    @classmethod
    def _comprehensive_bounds(cls, coordinate_system: str) -> "PhaseSpaceBounds":
        """Comprehensive bounds covering most asteroid populations."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(0.3, 5.5),
                lon=(0.0, 360.0),
                lat=(-90.0, 90.0),
                vrho=(-0.03, 0.03),
                vlon=(-3.0, 3.0),
                vlat=(-3.0, 3.0),
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-6.0, 6.0),
                y=(-6.0, 6.0),
                z=(-3.0, 3.0),
                vx=(-0.04, 0.04),
                vy=(-0.04, 0.04),
                vz=(-0.03, 0.03),
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(0.3, 5.5), e=(0.0, 0.9), i=(0.0, 90.0), raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    @classmethod
    def _inner_main_belt_bounds(cls, coordinate_system: str) -> "PhaseSpaceBounds":
        """Inner main belt asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(1.8, 2.8),
                lon=(0.0, 360.0),
                lat=(-20.0, 20.0),
                vrho=(-0.008, 0.008),
                vlon=(-0.8, 0.8),
                vlat=(-0.8, 0.8),
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-3.0, 3.0),
                y=(-3.0, 3.0),
                z=(-0.8, 0.8),
                vx=(-0.015, 0.015),
                vy=(-0.015, 0.015),
                vz=(-0.008, 0.008),
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(1.8, 2.8), e=(0.0, 0.25), i=(0.0, 20.0), raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    @property
    def coordinate_names(self) -> list[str]:
        """Get list of coordinate names."""
        return list(self.bounds.keys())

    @property
    def bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds as a dictionary."""
        return self.bounds.copy()

    @property
    def ranges(self) -> np.ndarray:
        """Get coordinate ranges as numpy array."""
        return np.array([max_val - min_val for min_val, max_val in self.bounds.values()])

    @property
    def volume(self) -> float:
        """Calculate total 6D hypervolume."""
        return np.prod(self.ranges)

    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a 6D point lies within the bounds.

        Parameters
        ----------
        point : np.ndarray
            6D point in the coordinate system of these bounds

        Returns
        -------
        bool
            True if point is within bounds
        """
        if len(point) != len(self.bounds):
            return False

        for i, (coord_name, (min_val, max_val)) in enumerate(self.bounds.items()):
            if not (min_val <= point[i] <= max_val):
                return False
        return True


class OrbitVolumes(qv.Table):
    """
    Represents phase space volumes associated with test orbits.

    This class stores volume information for multiple test orbits,
    defining the phase space regions each orbit is expected to cover.

    The 6D arrays are interpreted based on coordinate_system:
    - spherical: [rho, lon, lat, vrho, vlon, vlat]
    - cartesian: [x, y, z, vx, vy, vz]
    - keplerian: [a, e, i, raan, ap, M]
    """

    orbit_id = qv.LargeStringColumn()
    coordinate_system = qv.LargeStringColumn()
    centers = qv.LargeListColumn(pa.float64())  # 6D center coordinates
    half_widths = qv.LargeListColumn(pa.float64())  # 6D half-widths
    volume = qv.Float64Column()  # Hypervolume

    def get_centers_array(self) -> np.ndarray:
        """Get centers as (N, 6) array."""
        if len(self) == 0:
            return np.array([]).reshape(0, 6)

        # Convert list column to numpy array
        centers_list = self.centers.to_numpy(zero_copy_only=False)
        return np.array([list(center) for center in centers_list])

    def get_half_widths_array(self) -> np.ndarray:
        """Get half-widths as (N, 6) array."""
        if len(self) == 0:
            return np.array([]).reshape(0, 6)

        # Convert list column to numpy array
        half_widths_list = self.half_widths.to_numpy(zero_copy_only=False)
        return np.array([list(hw) for hw in half_widths_list])

    def get_bounds(self, index: int) -> np.ndarray:
        """Get volume bounds for a specific orbit as [[min0, max0], [min1, max1], ..., [min5, max5]]"""
        center = list(self.centers[index].as_py())
        half_width = list(self.half_widths[index].as_py())
        bounds = np.zeros((6, 2))
        bounds[:, 0] = np.array(center) - np.array(half_width)  # min
        bounds[:, 1] = np.array(center) + np.array(half_width)  # max
        return bounds

    def contains_point(self, index: int, point: np.ndarray) -> bool:
        """Check if a 6D point lies within the volume at the given index."""
        if len(point) != 6:
            return False
        center = np.array(list(self.centers[index].as_py()))
        half_width = np.array(list(self.half_widths[index].as_py()))
        return np.all(np.abs(point - center) <= half_width)

    def intersects(self, index1: int, index2: int) -> bool:
        """Check if two volumes intersect."""
        center1 = np.array(list(self.centers[index1].as_py()))
        half_width1 = np.array(list(self.half_widths[index1].as_py()))
        center2 = np.array(list(self.centers[index2].as_py()))
        half_width2 = np.array(list(self.half_widths[index2].as_py()))

        for i in range(6):
            if (
                center1[i] + half_width1[i] < center2[i] - half_width2[i]
                or center1[i] - half_width1[i] > center2[i] + half_width2[i]
            ):
                return False
        return True

    def intersection_volume(self, index1: int, index2: int) -> float:
        """Compute volume of intersection between two volumes."""
        if not self.intersects(index1, index2):
            return 0.0

        center1 = np.array(list(self.centers[index1].as_py()))
        half_width1 = np.array(list(self.half_widths[index1].as_py()))
        center2 = np.array(list(self.centers[index2].as_py()))
        half_width2 = np.array(list(self.half_widths[index2].as_py()))

        overlap_widths = np.zeros(6)
        for i in range(6):
            min_bound = max(center1[i] - half_width1[i], center2[i] - half_width2[i])
            max_bound = min(center1[i] + half_width1[i], center2[i] + half_width2[i])
            overlap_widths[i] = max(0, max_bound - min_bound)

        return np.prod(overlap_widths)

    @classmethod
    def from_arrays(
        cls, orbit_ids: List[str], coordinate_system: str, centers: np.ndarray, half_widths: np.ndarray
    ) -> "OrbitVolumes":
        """
        Create OrbitVolumes from numpy arrays.

        Parameters
        ----------
        orbit_ids : List[str]
            List of orbit IDs
        coordinate_system : str
            Coordinate system ("spherical", "cartesian", "keplerian")
        centers : np.ndarray
            (N, 6) array of volume centers
        half_widths : np.ndarray
            (N, 6) array of volume half-widths

        Returns
        -------
        OrbitVolumes
            New OrbitVolumes instance
        """
        n_points = len(orbit_ids)
        centers_list = [centers[i, :].tolist() for i in range(n_points)]
        half_widths_list = [half_widths[i, :].tolist() for i in range(n_points)]
        volumes = [np.prod(2 * half_widths[i, :]) for i in range(n_points)]

        return cls.from_kwargs(
            orbit_id=orbit_ids,
            coordinate_system=[coordinate_system] * n_points,
            centers=centers_list,
            half_widths=half_widths_list,
            volume=volumes,
        )

    # Convenience properties for different coordinate systems
    @property
    def spherical_coords(self) -> dict:
        """Get spherical coordinate values (only valid if coordinate_system == 'spherical')."""
        if len(self) == 0 or self.coordinate_system[0].as_py() != "spherical":
            raise ValueError("This OrbitVolumes is not in spherical coordinates")

        return {
            "rho": self.coord_0_center.to_numpy(zero_copy_only=False),
            "lon": self.coord_1_center.to_numpy(zero_copy_only=False),
            "lat": self.coord_2_center.to_numpy(zero_copy_only=False),
            "vrho": self.coord_3_center.to_numpy(zero_copy_only=False),
            "vlon": self.coord_4_center.to_numpy(zero_copy_only=False),
            "vlat": self.coord_5_center.to_numpy(zero_copy_only=False),
        }

    @property
    def cartesian_coords(self) -> dict:
        """Get cartesian coordinate values (only valid if coordinate_system == 'cartesian')."""
        if len(self) == 0 or self.coordinate_system[0].as_py() != "cartesian":
            raise ValueError("This OrbitVolumes is not in cartesian coordinates")

        return {
            "x": self.coord_0_center.to_numpy(zero_copy_only=False),
            "y": self.coord_1_center.to_numpy(zero_copy_only=False),
            "z": self.coord_2_center.to_numpy(zero_copy_only=False),
            "vx": self.coord_3_center.to_numpy(zero_copy_only=False),
            "vy": self.coord_4_center.to_numpy(zero_copy_only=False),
            "vz": self.coord_5_center.to_numpy(zero_copy_only=False),
        }

    @property
    def keplerian_coords(self) -> dict:
        """Get keplerian coordinate values (only valid if coordinate_system == 'keplerian')."""
        if len(self) == 0 or self.coordinate_system[0].as_py() != "keplerian":
            raise ValueError("This OrbitVolumes is not in keplerian coordinates")

        return {
            "a": self.coord_0_center.to_numpy(zero_copy_only=False),
            "e": self.coord_1_center.to_numpy(zero_copy_only=False),
            "i": self.coord_2_center.to_numpy(zero_copy_only=False),
            "raan": self.coord_3_center.to_numpy(zero_copy_only=False),
            "ap": self.coord_4_center.to_numpy(zero_copy_only=False),
            "M": self.coord_5_center.to_numpy(zero_copy_only=False),
        }


def _attach_volume_covariances(
    original_coords,
    half_widths: np.ndarray,
    coordinate_system: str,
    covariance_scale: float = 1.0,
):
    """
    Attach covariance matrices to coordinates based on volume half-widths, then convert to Cartesian.

    This function converts the volume half-widths (defined in the original coordinate system)
    into covariance matrices. The coordinate conversion to Cartesian will automatically
    transform the covariances using the appropriate Jacobian.

    Parameters
    ----------
    original_coords : Coordinates
        Original coordinates in the coordinate system where half-widths are defined
    half_widths : np.ndarray
        Half-widths for each dimension (6,)
    coordinate_system : str
        The coordinate system of the half-widths
    covariance_scale : float
        Scale factor for converting half-widths to sigmas (default: 1.0 = 1-sigma)

    Returns
    -------
    CartesianCoordinates
        Cartesian coordinates with attached covariances
    """
    from adam_core.coordinates import CoordinateCovariances

    n_points = len(original_coords)

    # Convert half-widths to sigmas (standard deviations)
    # half_widths are treated as 1-sigma by default, scale if needed
    sigmas = half_widths / covariance_scale

    # Create sigmas array (N, 6) - same sigmas for all orbits
    sigmas_array = np.tile(sigmas, (n_points, 1))

    # Create covariances from sigmas (creates diagonal covariance matrices)
    covariances = CoordinateCovariances.from_sigmas(sigmas_array)

    # Attach covariances to the original coordinates
    coords_with_cov = original_coords.set_column("covariance", covariances)

    # Convert to Cartesian - this will automatically transform the covariances!
    from adam_core.coordinates import CartesianCoordinates

    if not isinstance(coords_with_cov, CartesianCoordinates):
        cartesian_with_cov = coords_with_cov.to_cartesian()
    else:
        cartesian_with_cov = coords_with_cov

    return cartesian_with_cov


def generate_even_coverage_test_orbits(
    n_orbits: int,
    half_widths: Optional[np.ndarray] = None,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    attach_covariances: bool = True,
    covariance_scale: float = 1.0,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate test orbits with even phase space coverage using uniform grid spacing.

    This function creates test orbits distributed evenly across 6D orbital
    phase space using a uniform grid (same number of points in each dimension).
    This avoids issues with mixing different units and provides symmetric,
    predictable sampling. For custom grid spacing, use generate_custom_grid_test_orbits().

    Parameters
    ----------
    n_orbits : int
        Number of test orbits to generate
    half_widths : np.ndarray, optional
        Volume half-widths for the 6 coordinates in the chosen system.
        Default values depend on coordinate_system:
        - spherical: [0.1 AU, 15°, 8°, 0.002 AU/day, 0.2°/day, 0.15°/day]
        - cartesian: [0.1 AU, 0.1 AU, 0.1 AU, 0.002 AU/day, 0.002 AU/day, 0.002 AU/day]
        - keplerian: [0.1 AU, 0.01, 5°, 10°, 10°, 10°]
    bounds : Dict[str, Tuple[float, float]] or PhaseSpaceBounds, optional
        Phase space bounds for the chosen coordinate system. Can be a dictionary
        {'coord': (min, max)} or a PhaseSpaceBounds object. If None, uses asteroid_type.
    coordinate_system : str, optional
        Coordinate system for gridding and analysis: "spherical", "cartesian", or "keplerian".
        Default: "spherical"
    asteroid_type : str, optional
        Type of asteroid population for default bounds: "main_belt", "near_earth",
        "jupiter_trojans", "comprehensive", or "inner_main_belt".
        Used only if bounds is None. Default: "comprehensive"
    epoch : Timestamp, optional
        Epoch for the orbits. Default: J2000.0
    frame : str, optional
        Coordinate frame. Default: "ecliptic"
    attach_covariances : bool, optional
        If True, attach covariance matrices to the test orbit coordinates based on the
        volume half-widths. Default: True
    covariance_scale : float, optional
        Scale factor for converting half-widths to covariance sigmas. Default: 1.0
        (treats half-widths as 1-sigma uncertainties). Use 3.0 if half-widths
        represent 3-sigma confidence intervals.

    Returns
    -------
    test_orbits : TestOrbits
        Generated test orbits with even phase space distribution
    orbit_volumes : OrbitVolumes
        Volume information for each orbit, including center, half-widths, and volume size (quivr Table)
    report : Dict
        Coverage analysis report containing:
        - coverage_percentage: Fraction of phase space covered
        - overlap_percentage: Fraction of volume that overlaps
        - n_overlapping_pairs: Number of overlapping orbit pairs
        - efficiency: Coverage efficiency (1 - redundancy)
        - volume_stats: Statistics about individual volumes
        - coordinate_system: Which coordinate system was used
    """
    logger.info(
        f"Generating {n_orbits} test orbits with even phase space coverage in {coordinate_system} coordinates"
    )

    # Validate coordinate system
    if coordinate_system not in ["spherical", "cartesian", "keplerian"]:
        raise ValueError(
            f"coordinate_system must be 'spherical', 'cartesian', or 'keplerian', got '{coordinate_system}'"
        )

    # Set defaults based on coordinate system
    if half_widths is None:
        if coordinate_system == "spherical":
            half_widths = np.array([0.1, 15.0, 8.0, 0.002, 0.2, 0.15])  # rho, lon, lat, vrho, vlon, vlat
        elif coordinate_system == "cartesian":
            half_widths = np.array([0.1, 0.1, 0.1, 0.002, 0.002, 0.002])  # x, y, z, vx, vy, vz
        elif coordinate_system == "keplerian":
            half_widths = np.array([0.1, 0.01, 5.0, 10.0, 10.0, 10.0])  # a, e, i, raan, ap, M

    # Handle bounds - can be dict, PhaseSpaceBounds object, or None
    if bounds is None:

        # Use asteroid_type to get default bounds
        if asteroid_type is None:
            asteroid_type = "comprehensive"  # Cover whole asteroid population by default

        bounds_obj = PhaseSpaceBounds.from_type(asteroid_type, coordinate_system)
        bounds = bounds_obj.bounds_dict
    elif hasattr(bounds, "bounds_dict"):
        # It's a PhaseSpaceBounds object
        if bounds.coordinate_system != coordinate_system:
            raise ValueError(
                f"PhaseSpaceBounds coordinate system ({bounds.coordinate_system}) "
                f"doesn't match requested coordinate_system ({coordinate_system})"
            )
        bounds = bounds.bounds_dict
    else:
        # It's already a dictionary
        pass

    if epoch is None:
        epoch = Timestamp.from_jd([2451545.0], scale="tdb")  # J2000.0

    # Generate evenly distributed points in 6D phase space
    coords_6d = _generate_even_grid_points(n_orbits, bounds, coordinate_system)

    # Create coordinates in the specified system
    n_points = len(coords_6d)

    # Create repeated timestamps and origins for all points
    epoch_values = [epoch.jd().to_pylist()[0]] * n_points
    repeated_epoch = Timestamp.from_jd(epoch_values, scale=epoch.scale)
    repeated_origin = Origin.from_kwargs(code=["SUN"] * n_points)

    # Create coordinates based on the chosen system
    coordinates = _create_coordinates(coords_6d, coordinate_system, repeated_epoch, repeated_origin, frame)

    # Convert to Cartesian coordinates (required by TestOrbits)
    from adam_core.coordinates import CartesianCoordinates

    if attach_covariances:
        # Attach covariances in original coordinate system, then convert to Cartesian
        # (conversion will transform covariances automatically)
        cartesian_coords = _attach_volume_covariances(
            coordinates,
            half_widths,
            coordinate_system,
            covariance_scale,
        )
    else:
        # Just convert to Cartesian without covariances
        if not isinstance(coordinates, CartesianCoordinates):
            cartesian_coords = coordinates.to_cartesian()
        else:
            cartesian_coords = coordinates

    # Generate orbit IDs
    orbit_ids = [f"even_coverage_{i:04d}" for i in range(n_points)]

    # Create TestOrbits
    test_orbits = TestOrbits.from_kwargs(
        orbit_id=orbit_ids,
        coordinates=cartesian_coords,
    )

    # Create OrbitVolumes table
    individual_volume = np.prod(2 * half_widths)

    # Prepare data for OrbitVolumes table
    centers_list = [coords_6d[i, :] for i in range(n_points)]
    half_widths_list = [half_widths for _ in range(n_points)]

    orbit_volumes_data = {
        "orbit_id": orbit_ids,
        "coordinate_system": [coordinate_system] * n_points,
        "centers": centers_list,
        "half_widths": half_widths_list,
        "volume": np.full(n_points, individual_volume),
    }

    orbit_volumes = OrbitVolumes.from_kwargs(**orbit_volumes_data)

    # Analyze coverage (optional)
    # Create basic report with essential coverage metrics
    # For detailed diagnostics, use analyze_orbit_coverage_diagnostics() separately
    report = _create_basic_report(coords_6d, half_widths, bounds, coordinate_system)
    logger.info(f"Generated {len(test_orbits)} test orbits")
    logger.info(f"Coverage: {report['coverage_percentage']:.1f}%")

    return test_orbits, orbit_volumes, report


def _generate_even_grid_points(
    n_orbits: int, bounds: Dict[str, Tuple[float, float]], coordinate_system: str
) -> np.ndarray:
    """
    Generate evenly distributed points in 6D phase space using a uniform grid approach.

    Creates a uniform grid where each dimension gets the same number of grid points,
    avoiding issues with mixing different units (AU vs degrees). This provides
    symmetric, predictable sampling across all dimensions.

    Parameters
    ----------
    n_orbits : int
        Target number of orbits
    bounds : Dict[str, Tuple[float, float]]
        Phase space bounds
    coordinate_system : str
        Coordinate system being used

    Returns
    -------
    np.ndarray
        Array of shape (n_actual, 6) with coordinates
    """
    # Get coordinate names based on system
    if coordinate_system == "spherical":
        coord_names = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "raan", "ap", "M"]

    # Calculate uniform grid dimensions to get as close as possible to n_orbits
    # Start with 6th root as base for uniform grid
    base_n = max(1, int(np.ceil(n_orbits ** (1 / 6))))

    # Start with uniform grid
    grid_dims = np.full(6, base_n)
    total_points = np.prod(grid_dims)

    # If we're under target, try incrementing dimensions one by one
    if total_points < n_orbits:
        # Try increasing each dimension by 1 and see which gets closest
        for _ in range(10):  # Max 10 adjustments to avoid infinite loops
            best_dim = -1
            best_total = total_points

            # Try incrementing each dimension
            for dim in range(6):
                test_dims = grid_dims.copy()
                test_dims[dim] += 1
                test_total = np.prod(test_dims)

                # Pick the increment that gets closest to target without going too far over
                if test_total <= n_orbits * 1.2:  # Allow 20% overshoot
                    if abs(test_total - n_orbits) < abs(best_total - n_orbits):
                        best_total = test_total
                        best_dim = dim

            # Apply the best increment
            if best_dim >= 0:
                grid_dims[best_dim] += 1
                total_points = np.prod(grid_dims)
            else:
                break  # No good increments found

    logger.info(f"Using uniform grid dimensions: {grid_dims} (total: {total_points} points)")

    # Generate grid points
    grid_coords = []
    for i, coord in enumerate(coord_names):
        min_val, max_val = bounds[coord]
        n_points = grid_dims[i]

        if n_points == 1:
            coords = np.array([(min_val + max_val) / 2])
        else:
            # Add small margin to avoid boundary issues
            margin = (max_val - min_val) * 0.01
            coords = np.linspace(min_val + margin, max_val - margin, n_points)

        grid_coords.append(coords)

    # Create meshgrid and flatten
    mesh = np.meshgrid(*grid_coords, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])

    # Handle cases where we have too many or too few points
    if len(points) > n_orbits:
        # Too many points - randomly sample to get exactly n_orbits
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(points), size=n_orbits, replace=False)
        points = points[indices]
        logger.info(f"Sampled {n_orbits} points from {len(mesh[0].ravel())} grid points")
    elif len(points) < n_orbits:
        # Too few points - add random points within bounds to reach target
        n_additional = n_orbits - len(points)
        np.random.seed(42)  # For reproducibility

        additional_points = []
        for i, coord in enumerate(coord_names):
            min_val, max_val = bounds[coord]
            # Add small margin
            margin = (max_val - min_val) * 0.01
            random_coords = np.random.uniform(min_val + margin, max_val - margin, n_additional)
            additional_points.append(random_coords)

        additional_points = np.column_stack(additional_points)
        points = np.vstack([points, additional_points])
        logger.info(f"Added {n_additional} random points to reach {n_orbits} total")

    return points


def generate_custom_grid_test_orbits(
    grid_dimensions: np.ndarray,
    half_widths: Optional[np.ndarray] = None,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    attach_covariances: bool = True,
    covariance_scale: float = 1.0,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate test orbits using custom grid dimensions for precise control over sampling density.

    This function creates test orbits on a regular 6D grid with user-specified dimensions
    for each coordinate. Unlike generate_even_coverage_test_orbits, this gives exact control
    over the number of grid points in each dimension.

    Parameters
    ----------
    grid_dimensions : np.ndarray
        Number of grid points in each dimension [dim0, dim1, dim2, dim3, dim4, dim5].
        For spherical: [n_rho, n_lon, n_lat, n_vrho, n_vlon, n_vlat]
        For cartesian: [n_x, n_y, n_z, n_vx, n_vy, n_vz]
        For keplerian: [n_a, n_e, n_i, n_raan, n_ap, n_M]
        Total orbits = product of all dimensions.
    half_widths : np.ndarray, optional
        Volume half-widths for the 6 coordinates in the chosen system.
        Default values depend on coordinate_system:
        - spherical: [0.1 AU, 15°, 8°, 0.002 AU/day, 0.2°/day, 0.15°/day]
        - cartesian: [0.1 AU, 0.1 AU, 0.1 AU, 0.002 AU/day, 0.002 AU/day, 0.002 AU/day]
        - keplerian: [0.1 AU, 0.01, 5°, 10°, 10°, 10°]
    bounds : Dict[str, Tuple[float, float]] or PhaseSpaceBounds, optional
        Phase space bounds for the chosen coordinate system. Can be a dictionary
        {'coord': (min, max)} or a PhaseSpaceBounds object. If None, uses asteroid_type.
    coordinate_system : str, optional
        Coordinate system for gridding and analysis: "spherical", "cartesian", or "keplerian".
        Default: "spherical"
    asteroid_type : str, optional
        Type of asteroid population for default bounds: "main_belt", "near_earth",
        "jupiter_trojans", "comprehensive", or "inner_main_belt".
        Used only if bounds is None. Default: "comprehensive"
    epoch : Timestamp, optional
        Epoch for the orbits. Default: J2000.0
    frame : str, optional
        Coordinate frame. Default: "ecliptic"
    attach_covariances : bool, optional
        If True, attach covariance matrices to the test orbit coordinates based on the
        volume half-widths. Default: True
    covariance_scale : float, optional
        Scale factor for converting half-widths to covariance sigmas. Default: 1.0
        (treats half-widths as 1-sigma uncertainties). Use 3.0 if half-widths
        represent 3-sigma confidence intervals.

    Returns
    -------
    test_orbits : TestOrbits
        Generated test orbits on regular grid
    orbit_volumes : OrbitVolumes
        Volume information for each orbit, including center, half-widths, and volume size
    report : Dict
        Basic coverage report containing:
        - n_orbits: Number of orbits generated
        - coverage_percentage: Fraction of phase space covered
        - grid_dimensions: The grid dimensions used
        - coordinate_system: Which coordinate system was used

    Examples
    --------
    >>> # 2x more spatial resolution than velocity
    >>> grid_dims = np.array([10, 20, 20, 5, 5, 5])  # 200k orbits total
    >>> orbits, volumes, report = generate_custom_grid_test_orbits(grid_dims)

    >>> # High longitude resolution for sky surveys
    >>> grid_dims = np.array([5, 50, 10, 3, 3, 3])  # 67.5k orbits
    >>> orbits, volumes, report = generate_custom_grid_test_orbits(
    ...     grid_dims, coordinate_system="spherical"
    ... )
    """
    logger.info(f"Generating custom grid test orbits in {coordinate_system} coordinates")
    logger.info(f"Grid dimensions: {grid_dimensions} (total: {np.prod(grid_dimensions):,} orbits)")

    # Validate inputs
    if coordinate_system not in ["spherical", "cartesian", "keplerian"]:
        raise ValueError(
            f"coordinate_system must be 'spherical', 'cartesian', or 'keplerian', got '{coordinate_system}'"
        )

    if len(grid_dimensions) != 6:
        raise ValueError(f"grid_dimensions must have 6 elements, got {len(grid_dimensions)}")

    if np.any(np.array(grid_dimensions) < 1):
        raise ValueError(f"All grid dimensions must be >= 1, got {grid_dimensions}")

    # Ensure grid_dimensions is a numpy array for consistency
    grid_dimensions = np.array(grid_dimensions)
    
    # Ensure half_widths is a numpy array for consistency (if provided)
    if half_widths is not None:
        half_widths = np.array(half_widths)

    # Set defaults based on coordinate system
    if half_widths is None:
        if coordinate_system == "spherical":
            half_widths = np.array([0.1, 15.0, 8.0, 0.002, 0.2, 0.15])  # rho, lon, lat, vrho, vlon, vlat
        elif coordinate_system == "cartesian":
            half_widths = np.array([0.1, 0.1, 0.1, 0.002, 0.002, 0.002])  # x, y, z, vx, vy, vz
        elif coordinate_system == "keplerian":
            half_widths = np.array([0.1, 0.01, 5.0, 10.0, 10.0, 10.0])  # a, e, i, raan, ap, M

    # Handle bounds - can be dict, PhaseSpaceBounds object, or None
    if bounds is None:
        # Use asteroid_type to get default bounds
        if asteroid_type is None:
            asteroid_type = "comprehensive"  # Cover whole asteroid population by default

        bounds_obj = PhaseSpaceBounds.from_type(asteroid_type, coordinate_system)
        bounds = bounds_obj.bounds_dict
    elif hasattr(bounds, "bounds_dict"):
        # It's a PhaseSpaceBounds object
        if bounds.coordinate_system != coordinate_system:
            raise ValueError(
                f"PhaseSpaceBounds coordinate system ({bounds.coordinate_system}) "
                f"doesn't match requested coordinate_system ({coordinate_system})"
            )
        bounds = bounds.bounds_dict
    else:
        # It's already a dictionary
        pass

    if epoch is None:
        epoch = Timestamp.from_jd([2451545.0], scale="tdb")  # J2000.0

    # Generate grid points using custom dimensions
    coords_6d = _generate_custom_grid_points(grid_dimensions, bounds, coordinate_system)

    # Create coordinates in the specified system
    n_points = len(coords_6d)

    # Create repeated timestamps and origins for all points
    epoch_values = [epoch.jd().to_pylist()[0]] * n_points
    repeated_epoch = Timestamp.from_jd(epoch_values, scale=epoch.scale)
    repeated_origin = Origin.from_kwargs(code=["SUN"] * n_points)

    # Create coordinates based on the chosen system
    coordinates = _create_coordinates(coords_6d, coordinate_system, repeated_epoch, repeated_origin, frame)

    # Convert to Cartesian coordinates (required by TestOrbits)
    from adam_core.coordinates import CartesianCoordinates

    if attach_covariances:
        # Attach covariances in original coordinate system, then convert to Cartesian
        # (conversion will transform covariances automatically)
        cartesian_coords = _attach_volume_covariances(
            coordinates,
            half_widths,
            coordinate_system,
            covariance_scale,
        )
    else:
        # Just convert to Cartesian without covariances
        if not isinstance(coordinates, CartesianCoordinates):
            cartesian_coords = coordinates.to_cartesian()
        else:
            cartesian_coords = coordinates

    # Generate orbit IDs
    orbit_ids = [f"custom_grid_{i:04d}" for i in range(n_points)]

    # Create TestOrbits
    test_orbits = TestOrbits.from_kwargs(
        orbit_id=orbit_ids,
        coordinates=cartesian_coords,
    )

    # Create OrbitVolumes table
    individual_volume = np.prod(2 * half_widths)

    # Prepare data for OrbitVolumes table
    centers_list = [coords_6d[i, :] for i in range(n_points)]
    half_widths_list = [half_widths for _ in range(n_points)]

    orbit_volumes_data = {
        "orbit_id": orbit_ids,
        "coordinate_system": [coordinate_system] * n_points,
        "centers": centers_list,
        "half_widths": half_widths_list,
        "volume": np.full(n_points, individual_volume),
    }

    orbit_volumes = OrbitVolumes.from_kwargs(**orbit_volumes_data)

    # Create basic report with essential coverage metrics
    report = _create_basic_report(coords_6d, half_widths, bounds, coordinate_system)
    report["grid_dimensions"] = grid_dimensions
    report["n_orbits"] = n_points

    logger.info(f"Generated {len(test_orbits)} test orbits")
    logger.info(f"Coverage: {report['coverage_percentage']:.1f}%")

    return test_orbits, orbit_volumes, report


def _generate_custom_grid_points(
    grid_dimensions: np.ndarray, bounds: Dict[str, Tuple[float, float]], coordinate_system: str
) -> np.ndarray:
    """
    Generate grid points using custom dimensions for each coordinate.

    Parameters
    ----------
    grid_dimensions : np.ndarray
        Number of grid points in each dimension [dim0, dim1, ..., dim5]
    bounds : Dict[str, Tuple[float, float]]
        Phase space bounds
    coordinate_system : str
        Coordinate system being used

    Returns
    -------
    np.ndarray
        Array of shape (n_total, 6) with coordinates where n_total = product(grid_dimensions)
    """
    # Get coordinate names based on system
    if coordinate_system == "spherical":
        coord_names = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "raan", "ap", "M"]

    # Create 1D grids for each dimension
    grids_1d = []
    for i, coord in enumerate(coord_names):
        min_val, max_val = bounds[coord]
        n_points = int(grid_dimensions[i])

        if n_points == 1:
            # Single point at center
            grid_1d = np.array([(min_val + max_val) / 2])
        else:
            # Regular grid from min to max
            grid_1d = np.linspace(min_val, max_val, n_points)

        grids_1d.append(grid_1d)

    # Create 6D meshgrid
    mesh = np.meshgrid(*grids_1d, indexing="ij")

    # Flatten and combine
    points = np.column_stack([m.ravel() for m in mesh])

    total_points = np.prod(grid_dimensions)
    logger.info(f"Generated {len(points)} grid points (expected: {total_points})")

    return points


def get_sampling_weights_preset(preset: str, coordinate_system: str = "spherical") -> np.ndarray:
    """
    Get predefined sampling weight presets for common use cases.

    Parameters
    ----------
    preset : str
        Preset name:
        - "uniform": Equal sampling in all dimensions
        - "spatial_focus": 2x more sampling in spatial dimensions
        - "spatial_heavy": 3x more sampling in spatial dimensions
        - "velocity_focus": 2x more sampling in velocity dimensions
        - "position_only": Heavy focus on position, minimal velocity
    coordinate_system : str, optional
        Coordinate system for dimension interpretation. Default: "spherical"

    Returns
    -------
    np.ndarray
        Sampling weights array [dim0, dim1, dim2, dim3, dim4, dim5]

    Examples
    --------
    >>> # Generate orbits with 2x spatial resolution
    >>> weights = get_sampling_weights_preset("spatial_focus")
    >>> orbits, volumes, report = generate_even_coverage_test_orbits(
    ...     n_orbits=1000000, sampling_weights=weights
    ... )
    """
    presets = {
        "uniform": [1, 1, 1, 1, 1, 1],
        "spatial_focus": [2, 2, 2, 1, 1, 1],
        "spatial_heavy": [3, 3, 3, 1, 1, 1],
        "velocity_focus": [1, 1, 1, 2, 2, 2],
        "position_only": [4, 4, 4, 0.5, 0.5, 0.5],
    }

    if preset not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    weights = np.array(presets[preset], dtype=float)

    # Log what the weights mean for the coordinate system
    if coordinate_system == "spherical":
        coord_names = ["ρ", "lon", "lat", "vρ", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "Ω", "ω", "M"]
    else:
        coord_names = ["dim0", "dim1", "dim2", "dim3", "dim4", "dim5"]

    logger.info(f"Sampling weights preset '{preset}' for {coordinate_system}:")
    for i, (name, weight) in enumerate(zip(coord_names, weights)):
        logger.info(f"  {name}: {weight:.1f}x")

    return weights


def create_grid_dimensions(
    spatial_resolution: int = 10, velocity_resolution: int = 5, coordinate_system: str = "spherical"
) -> np.ndarray:
    """
    Create grid dimensions with different spatial and velocity resolutions.

    This is a convenience function for the common case of wanting more spatial
    resolution than velocity resolution.

    Parameters
    ----------
    spatial_resolution : int, optional
        Number of grid points for spatial coordinates. Default: 10
    velocity_resolution : int, optional
        Number of grid points for velocity coordinates. Default: 5
    coordinate_system : str, optional
        Coordinate system to determine which dimensions are spatial vs velocity.
        Default: "spherical"

    Returns
    -------
    np.ndarray
        Grid dimensions array suitable for generate_custom_grid_test_orbits()

    Examples
    --------
    >>> # 2x more spatial resolution
    >>> grid_dims = create_grid_dimensions(spatial_resolution=20, velocity_resolution=10)
    >>> # Results in [20, 20, 20, 10, 10, 10] for spherical coordinates

    >>> # High spatial focus for sky surveys
    >>> grid_dims = create_grid_dimensions(spatial_resolution=50, velocity_resolution=3)
    >>> # Results in [50, 50, 50, 3, 3, 3] = 1.125M orbits
    """
    if coordinate_system in ["spherical", "cartesian"]:
        # First 3 are spatial, last 3 are velocity
        grid_dims = np.array(
            [
                spatial_resolution,
                spatial_resolution,
                spatial_resolution,
                velocity_resolution,
                velocity_resolution,
                velocity_resolution,
            ]
        )
    elif coordinate_system == "keplerian":
        # For Keplerian, a/e/i are more "spatial", angles/M are more "velocity-like"
        grid_dims = np.array(
            [
                spatial_resolution,
                spatial_resolution,
                spatial_resolution,
                velocity_resolution,
                velocity_resolution,
                velocity_resolution,
            ]
        )
    else:
        raise ValueError(f"Unknown coordinate_system: {coordinate_system}")

    total_orbits = np.prod(grid_dims)
    logger.info(f"Created grid dimensions {grid_dims} for {coordinate_system}")
    logger.info(f"Total orbits: {total_orbits:,}")

    return grid_dims


def create_custom_grid_dimensions(
    rho_or_x: int = 10,
    lon_or_y: int = 10,
    lat_or_z: int = 10,
    vrho_or_vx: int = 5,
    vlon_or_vy: int = 5,
    vlat_or_vz: int = 5,
    coordinate_system: str = "spherical",
) -> np.ndarray:
    """
    Create fully custom grid dimensions with individual control over each coordinate.

    Parameters
    ----------
    rho_or_x : int, optional
        Grid points for first coordinate (ρ/x/a). Default: 10
    lon_or_y : int, optional
        Grid points for second coordinate (lon/y/e). Default: 10
    lat_or_z : int, optional
        Grid points for third coordinate (lat/z/i). Default: 10
    vrho_or_vx : int, optional
        Grid points for fourth coordinate (vρ/vx/Ω). Default: 5
    vlon_or_vy : int, optional
        Grid points for fifth coordinate (vlon/vy/ω). Default: 5
    vlat_or_vz : int, optional
        Grid points for sixth coordinate (vlat/vz/M). Default: 5
    coordinate_system : str, optional
        Coordinate system for labeling. Default: "spherical"

    Returns
    -------
    np.ndarray
        Grid dimensions array [dim0, dim1, dim2, dim3, dim4, dim5]

    Examples
    --------
    >>> # High longitude resolution for sky surveys
    >>> grid_dims = create_custom_grid_dimensions(
    ...     rho_or_x=5, lon_or_y=100, lat_or_z=20,
    ...     vrho_or_vx=3, vlon_or_vy=3, vlat_or_vz=3
    ... )  # 5×100×20×3×3×3 = 270k orbits

    >>> # Focus on semi-major axis and eccentricity for Keplerian
    >>> grid_dims = create_custom_grid_dimensions(
    ...     rho_or_x=50, lon_or_y=20, lat_or_z=5,
    ...     vrho_or_vx=5, vlon_or_vy=5, vlat_or_vz=5,
    ...     coordinate_system="keplerian"
    ... )  # 50×20×5×5×5×5 = 1.25M orbits
    """
    grid_dims = np.array([rho_or_x, lon_or_y, lat_or_z, vrho_or_vx, vlon_or_vy, vlat_or_vz])

    # Get coordinate names for logging
    if coordinate_system == "spherical":
        coord_names = ["ρ", "lon", "lat", "vρ", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "Ω", "ω", "M"]
    else:
        coord_names = ["dim0", "dim1", "dim2", "dim3", "dim4", "dim5"]

    total_orbits = np.prod(grid_dims)
    logger.info(f"Created custom grid dimensions for {coordinate_system}:")
    for name, dim in zip(coord_names, grid_dims):
        logger.info(f"  {name}: {dim} points")
    logger.info(f"Total orbits: {total_orbits:,}")

    return grid_dims


def _create_coordinates(
    coords_6d: np.ndarray, coordinate_system: str, time: Timestamp, origin: Origin, frame: str
):
    """
    Create coordinate objects based on the specified system.

    Parameters
    ----------
    coords_6d : np.ndarray
        6D coordinate values
    coordinate_system : str
        Type of coordinate system
    time : Timestamp
        Time for coordinates
    origin : Origin
        Origin for coordinates
    frame : str
        Reference frame

    Returns
    -------
    Coordinates object (SphericalCoordinates, CartesianCoordinates, or KeplerianCoordinates)
    """
    from adam_core.coordinates import CartesianCoordinates, KeplerianCoordinates

    if coordinate_system == "spherical":
        return SphericalCoordinates.from_kwargs(
            rho=coords_6d[:, 0],
            lon=coords_6d[:, 1],
            lat=coords_6d[:, 2],
            vrho=coords_6d[:, 3],
            vlon=coords_6d[:, 4],
            vlat=coords_6d[:, 5],
            time=time,
            origin=origin,
            frame=frame,
        )
    elif coordinate_system == "cartesian":
        return CartesianCoordinates.from_kwargs(
            x=coords_6d[:, 0],
            y=coords_6d[:, 1],
            z=coords_6d[:, 2],
            vx=coords_6d[:, 3],
            vy=coords_6d[:, 4],
            vz=coords_6d[:, 5],
            time=time,
            origin=origin,
            frame=frame,
        )
    elif coordinate_system == "keplerian":
        return KeplerianCoordinates.from_kwargs(
            a=coords_6d[:, 0],
            e=coords_6d[:, 1],
            i=coords_6d[:, 2],
            raan=coords_6d[:, 3],
            ap=coords_6d[:, 4],
            M=coords_6d[:, 5],
            time=time,
            origin=origin,
            frame=frame,
        )
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def _analyze_coverage(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
) -> Dict:
    """
    Analyze phase space coverage and overlaps.

    Parameters
    ----------
    coords : np.ndarray
        Orbit coordinates (n_orbits, 6)
    half_widths : np.ndarray
        Volume half-widths (6,)
    bounds : Dict[str, Tuple[float, float]]
        Phase space bounds

    Returns
    -------
    Dict
        Coverage analysis report
    """
    n_orbits = len(coords)

    # Calculate individual volume
    individual_volume = np.prod(2 * half_widths)

    # Calculate total phase space volume
    if coordinate_system == "spherical":
        coord_names = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "raan", "ap", "M"]

    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    total_phase_space = np.prod(ranges)

    # Estimate coverage (simple approximation)
    total_volume_no_overlap = n_orbits * individual_volume
    coverage_percentage = min(100.0, 100.0 * total_volume_no_overlap / total_phase_space)

    # Analyze overlaps (simplified - check center distances)
    overlap_count = 0
    total_overlap_volume = 0.0

    # Fast overlap analysis - use sampling for large datasets
    if n_orbits > 1000:
        # For large datasets, use statistical sampling to estimate overlaps
        overlap_count, total_overlap_volume = _fast_overlap_analysis(coords, half_widths, n_orbits)
    else:
        # For smaller datasets, do exact calculation with vectorized operations
        overlap_count, total_overlap_volume = _exact_overlap_analysis(coords, half_widths, n_orbits)

    # Calculate metrics
    overlap_percentage = (
        100.0 * total_overlap_volume / total_volume_no_overlap if total_volume_no_overlap > 0 else 0.0
    )
    efficiency = (
        1.0 - (total_overlap_volume / total_volume_no_overlap) if total_volume_no_overlap > 0 else 1.0
    )

    # Volume statistics
    volume_stats = {
        "individual_volume": individual_volume,
        "total_volume_no_overlap": total_volume_no_overlap,
        "total_overlap_volume": total_overlap_volume,
        "mean_center_distance": _calculate_mean_distance(coords) if n_orbits > 1 else 0.0,
    }

    return {
        "n_orbits": n_orbits,
        "coverage_percentage": coverage_percentage,
        "overlap_percentage": overlap_percentage,
        "n_overlapping_pairs": overlap_count,
        "efficiency": efficiency,
        "volume_stats": volume_stats,
        "phase_space_volume": total_phase_space,
        "bounds": bounds,
        "half_widths": half_widths,
        "coordinate_system": coordinate_system,
    }


def generate_orbit_volumes_for_target_coverage(
    n_orbits: int,
    target_coverage_percent: float,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate test orbits with volumes sized to achieve a target coverage percentage.

    This function calculates the required volume size directly from the target coverage
    and generates orbits with those volumes. Much faster than iterative approaches.

    Parameters
    ----------
    n_orbits : int
        Number of test orbits to generate
    target_coverage_percent : float
        Desired coverage percentage (0-100)
    bounds : Dict[str, Tuple[float, float]] or PhaseSpaceBounds, optional
        Phase space bounds for the chosen coordinate system. If None, uses asteroid_type.
    coordinate_system : str, optional
        Coordinate system: "spherical", "cartesian", or "keplerian". Default: "spherical"
    asteroid_type : str, optional
        Type of asteroid population for default bounds. Default: "main_belt"
    epoch : Timestamp, optional
        Epoch for the orbits. Default: J2000.0
    frame : str, optional
        Coordinate frame. Default: "ecliptic"

    Returns
    -------
    Tuple[TestOrbits, OrbitVolumes, Dict]
        - TestOrbits: Generated test orbits
        - OrbitVolumes: Volume information for each orbit (quivr Table)
        - Dict: Coverage analysis report with additional keys:
            - 'target_coverage_percent': Requested coverage percentage
            - 'actual_coverage_percent': Achieved coverage percentage
            - 'volume_calculation_method': 'direct' (non-iterative)
    """
    logger.info(f"Generating {n_orbits} orbits to achieve {target_coverage_percent:.1f}% coverage")

    # Validate inputs
    if not 0 < target_coverage_percent <= 100:
        raise ValueError("target_coverage_percent must be between 0 and 100")
    if n_orbits < 1:
        raise ValueError("n_orbits must be at least 1")

    # Get bounds
    if bounds is None:
        if asteroid_type is None:
            asteroid_type = "main_belt"
        bounds = PhaseSpaceBounds.from_type(asteroid_type, coordinate_system)
    elif isinstance(bounds, dict):
        bounds = PhaseSpaceBounds(coordinate_system=coordinate_system, bounds=bounds)

    # Direct calculation of required volume size (no iteration needed!)
    phase_space_volume = bounds.volume
    target_total_volume = (target_coverage_percent / 100.0) * phase_space_volume
    target_individual_volume = target_total_volume / n_orbits

    logger.info(f"Target total volume: {target_total_volume:.2e}")
    logger.info(f"Target individual volume: {target_individual_volume:.2e}")

    # Calculate half-widths to achieve target volume
    # For 6D hyperrectangle: volume = prod(2 * half_widths)
    # We'll scale proportionally to coordinate ranges for reasonable shapes
    coord_ranges = bounds.ranges

    # Calculate scale factor: if we scale all ranges by this factor, we get target volume
    # target_volume = prod(scale_factor * ranges) = scale_factor^6 * prod(ranges)
    # So: scale_factor = (target_volume / prod(ranges))^(1/6)
    range_volume = np.prod(coord_ranges)
    scale_factor = (target_individual_volume / range_volume) ** (1 / 6)

    # Half-widths are half the scaled ranges
    half_widths = scale_factor * coord_ranges / 2

    logger.info(f"Volume scale factor: {scale_factor:.4f}")
    logger.info(f"Half-widths: {half_widths}")

    # Generate orbits with calculated half-widths (single call!)
    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits=n_orbits,
        half_widths=half_widths,
        bounds=bounds,
        coordinate_system=coordinate_system,
        epoch=epoch,
        frame=frame,
    )

    # Add target coverage info to report
    actual_coverage = report["coverage_percentage"]
    report["target_coverage_percent"] = target_coverage_percent
    report["actual_coverage_percent"] = actual_coverage
    report["volume_calculation_method"] = "direct"
    report["coverage_error_percent"] = abs(actual_coverage - target_coverage_percent)

    logger.info(f"Target: {target_coverage_percent:.1f}%, Actual: {actual_coverage:.1f}%")
    logger.info(f"Error: {abs(actual_coverage - target_coverage_percent):.1f}%")

    return test_orbits, orbit_volumes, report


def generate_orbits_for_coverage_with_fixed_volumes(
    target_coverage_percent: float,
    half_widths: np.ndarray,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    max_orbits: int = 100000,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Calculate and generate the exact number of orbits needed to achieve target coverage with fixed volume sizes.

    This function calculates the required number of orbits directly from the target coverage
    and fixed volume sizes. Much faster than iterative approaches.

    Parameters
    ----------
    target_coverage_percent : float
        Desired coverage percentage (0-100)
    half_widths : np.ndarray
        Fixed volume half-widths for the 6 coordinates in the chosen system
    bounds : Dict[str, Tuple[float, float]] or PhaseSpaceBounds, optional
        Phase space bounds for the chosen coordinate system. If None, uses asteroid_type.
    coordinate_system : str, optional
        Coordinate system: "spherical", "cartesian", or "keplerian". Default: "spherical"
    asteroid_type : str, optional
        Type of asteroid population for default bounds. Default: "main_belt"
    epoch : Timestamp, optional
        Epoch for the orbits. Default: J2000.0
    frame : str, optional
        Coordinate frame. Default: "ecliptic"
    max_orbits : int, optional
        Maximum number of orbits to generate. Default: 100,000

    Returns
    -------
    Tuple[TestOrbits, OrbitVolumes, Dict]
        - TestOrbits: Generated test orbits
        - OrbitVolumes: Volume information for each orbit (quivr Table)
        - Dict: Coverage analysis report with additional keys:
            - 'target_coverage_percent': Requested coverage percentage
            - 'actual_coverage_percent': Achieved coverage percentage
            - 'calculation_method': 'direct' (non-iterative)
            - 'orbits_requested': Number of orbits calculated as needed
            - 'hit_max_orbits_limit': Whether calculation exceeded max_orbits
    """
    logger.info(f"Generating orbits to achieve {target_coverage_percent:.1f}% coverage with fixed volumes")

    # Validate inputs
    if not 0 < target_coverage_percent <= 100:
        raise ValueError("target_coverage_percent must be between 0 and 100")
    if len(half_widths) != 6:
        raise ValueError("half_widths must be 6-dimensional")
    if max_orbits < 1:
        raise ValueError("max_orbits must be at least 1")

    # Get bounds
    if bounds is None:
        if asteroid_type is None:
            asteroid_type = "main_belt"
        bounds = PhaseSpaceBounds.from_type(asteroid_type, coordinate_system)
    elif isinstance(bounds, dict):
        bounds = PhaseSpaceBounds(coordinate_system=coordinate_system, bounds=bounds)

    # Direct calculation of required number of orbits (no iteration needed!)
    individual_volume = np.prod(2 * half_widths)
    phase_space_volume = bounds.volume
    target_total_volume = (target_coverage_percent / 100.0) * phase_space_volume

    # Calculate exact number of orbits needed (assuming minimal overlap)
    n_orbits_needed = max(1, int(np.ceil(target_total_volume / individual_volume)))

    # Check if we exceed max_orbits limit
    if n_orbits_needed > max_orbits:
        logger.warning(
            f"Need {n_orbits_needed} orbits for {target_coverage_percent:.1f}% coverage, but limited to {max_orbits}"
        )
        n_orbits_needed = max_orbits

    logger.info(f"Individual volume: {individual_volume:.2e}")
    logger.info(f"Phase space volume: {phase_space_volume:.2e}")
    logger.info(f"Target total volume: {target_total_volume:.2e}")
    logger.info(f"Calculated orbits needed: {n_orbits_needed}")

    # Generate orbits with calculated count (single call!)
    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits=n_orbits_needed,
        half_widths=half_widths,
        bounds=bounds,
        coordinate_system=coordinate_system,
        epoch=epoch,
        frame=frame,
    )

    # Add target coverage info to report
    actual_coverage = report["coverage_percentage"]
    report["target_coverage_percent"] = target_coverage_percent
    report["actual_coverage_percent"] = actual_coverage
    report["calculation_method"] = "direct"
    report["coverage_error_percent"] = abs(actual_coverage - target_coverage_percent)
    report["orbits_requested"] = n_orbits_needed
    report["hit_max_orbits_limit"] = n_orbits_needed >= max_orbits

    logger.info(f"Target: {target_coverage_percent:.1f}%, Actual: {actual_coverage:.1f}%")
    logger.info(f"Error: {abs(actual_coverage - target_coverage_percent):.1f}%")

    return test_orbits, orbit_volumes, report


def _fast_overlap_analysis(coords: np.ndarray, half_widths: np.ndarray, n_orbits: int) -> Tuple[int, float]:
    """
    Fast statistical overlap analysis for large datasets using sampling.

    Instead of O(n²) pairwise comparisons, sample a subset and extrapolate.
    Reduces 100k orbits from 5 billion comparisons to ~10k samples.
    """
    # Sample size: use sqrt(n) pairs for statistical estimation
    max_samples = min(10000, n_orbits * 10)  # Cap at 10k samples for performance

    if n_orbits <= 100:
        # Small enough for exact calculation
        return _exact_overlap_analysis(coords, half_widths, n_orbits)

    # Random sampling of pairs
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(n_orbits, size=(max_samples, 2), replace=True)

    # Remove self-pairs
    valid_pairs = sample_indices[sample_indices[:, 0] != sample_indices[:, 1]]

    overlap_count_sample = 0
    total_overlap_volume_sample = 0.0

    # Vectorized overlap checking for sampled pairs
    for i, j in valid_pairs:
        # Check if volumes overlap using vectorized operations
        center_diff = np.abs(coords[i] - coords[j])
        overlap_threshold = 2 * half_widths

        # Check if all dimensions overlap
        if np.all(center_diff <= overlap_threshold):
            overlap_count_sample += 1
            # Calculate overlap volume
            overlap_widths = overlap_threshold - center_diff
            overlap_vol = np.prod(overlap_widths)
            total_overlap_volume_sample += overlap_vol

    # Extrapolate from sample to full dataset
    total_possible_pairs = n_orbits * (n_orbits - 1) // 2
    sample_pairs = len(valid_pairs)

    if sample_pairs > 0:
        scale_factor = total_possible_pairs / sample_pairs
        estimated_overlap_count = int(overlap_count_sample * scale_factor)
        estimated_total_overlap_volume = total_overlap_volume_sample * scale_factor
    else:
        estimated_overlap_count = 0
        estimated_total_overlap_volume = 0.0

    return estimated_overlap_count, estimated_total_overlap_volume


def _exact_overlap_analysis(coords: np.ndarray, half_widths: np.ndarray, n_orbits: int) -> Tuple[int, float]:
    """
    Exact overlap analysis for smaller datasets - optimized vectorized version.
    Still O(n²) but with vectorized NumPy operations for better performance.
    """
    overlap_count = 0
    total_overlap_volume = 0.0

    # Use the original nested loop approach but optimized
    for i in range(n_orbits):
        for j in range(i + 1, n_orbits):
            # Vectorized overlap check
            center_diff = np.abs(coords[i] - coords[j])
            overlap_threshold = 2 * half_widths

            # Check if all dimensions overlap
            if np.all(center_diff <= overlap_threshold):
                overlap_count += 1
                # Calculate overlap volume
                overlap_widths = overlap_threshold - center_diff
                overlap_vol = np.prod(overlap_widths)
                total_overlap_volume += overlap_vol

    return overlap_count, total_overlap_volume


def _calculate_mean_distance(coords: np.ndarray) -> float:
    """
    Calculate mean pairwise distance efficiently.
    For large datasets, use sampling to avoid O(n²) computation.
    """
    n_orbits = len(coords)

    if n_orbits <= 1000:
        # Small enough for exact calculation
        distances = []
        for i in range(n_orbits):
            for j in range(i + 1, n_orbits):
                distances.append(np.linalg.norm(coords[i] - coords[j]))
        return np.mean(distances) if distances else 0.0
    else:
        # Sample distances for large datasets
        np.random.seed(42)  # For reproducibility
        max_samples = min(1000, n_orbits)  # Sample up to 1000 pairs

        sample_indices = np.random.choice(n_orbits, size=(max_samples, 2), replace=True)
        valid_pairs = sample_indices[sample_indices[:, 0] != sample_indices[:, 1]]

        distances = []
        for i, j in valid_pairs:
            distances.append(np.linalg.norm(coords[i] - coords[j]))

        return np.mean(distances) if distances else 0.0


def calculate_coverage_percentage(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
    method: str = "fast",
    **kwargs
) -> float:
    """
    Calculate coverage percentage using different methods.
    
    Parameters
    ----------
    coords : np.ndarray
        Orbit coordinates (n_orbits, 6)
    half_widths : np.ndarray
        Half-widths for each dimension (6,)
    bounds : Dict[str, Tuple[float, float]]
        Phase space bounds for each coordinate
    coordinate_system : str
        Coordinate system ("spherical", "cartesian", "keplerian")
    method : str
        Coverage calculation method:
        - "fast": O(1) - assumes no overlap (current default)
        - "monte_carlo": O(n×samples) - samples phase space
        - "adaptive_grid": O(n×bins^6) - bins phase space adaptively
        - "exact": O(n²) - full pairwise overlap analysis
    **kwargs
        Additional parameters for specific methods:
        - n_samples: int (for monte_carlo, default=100000)
        - n_bins_per_dim: int (for adaptive_grid, default=20/50)
    
    Returns
    -------
    float
        Coverage percentage (0-100)
    """
    if method == "fast":
        return _calculate_coverage_fast(coords, half_widths, bounds, coordinate_system)
    elif method == "monte_carlo":
        n_samples = kwargs.get("n_samples", 100000)
        return _calculate_coverage_monte_carlo(coords, half_widths, bounds, coordinate_system, n_samples)
    elif method == "adaptive_grid":
        return _calculate_coverage_adaptive_grid(coords, half_widths, bounds, coordinate_system, **kwargs)
    elif method == "exact":
        return _calculate_coverage_exact(coords, half_widths, bounds, coordinate_system)
    else:
        raise ValueError(f"Unknown coverage method: {method}. Choose from: fast, monte_carlo, adaptive_grid, exact")


def _calculate_coverage_fast(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
) -> float:
    """
    Fast O(1) coverage calculation - assumes no overlap.
    
    This is the original method used for iterative functions.
    """
    n_orbits = len(coords)
    individual_volume = np.prod(2 * half_widths)

    # Get coordinate names based on system
    coord_names = _get_coordinate_names(coordinate_system)

    # Calculate total phase space volume
    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    total_phase_space = np.prod(ranges)

    # Basic coverage estimate (no overlap analysis needed)
    total_volume_no_overlap = n_orbits * individual_volume
    coverage_percentage = min(100.0, 100.0 * total_volume_no_overlap / total_phase_space)

    return coverage_percentage


def _calculate_coverage_monte_carlo(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
    n_samples: int = 100000
) -> float:
    """
    Monte Carlo coverage calculation - samples phase space randomly.
    
    Time complexity: O(n × samples)
    No assumptions about orbit positions or overlaps.
    """
    coord_names = _get_coordinate_names(coordinate_system)
    ranges = np.array([bounds[coord] for coord in coord_names])
    
    # Generate random sample points across phase space
    np.random.seed(42)  # Reproducible results
    sample_points = np.random.uniform(
        low=ranges[:, 0], 
        high=ranges[:, 1], 
        size=(n_samples, len(coord_names))
    )
    
    covered_count = 0
    
    # Check each sample point against all orbit volumes
    for point in sample_points:
        # Check if point is inside any orbit volume
        for orbit_center in coords:
            if np.all(np.abs(point - orbit_center) <= half_widths):
                covered_count += 1
                break  # Point is covered, move to next
    
    coverage_percentage = covered_count / n_samples * 100
    return coverage_percentage


def _calculate_coverage_adaptive_grid(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
    coarse_bins: int = 20,
    fine_bins: int = 50,
    coverage_threshold: float = 10.0
) -> float:
    """
    Adaptive grid coverage calculation - uses coarse grid, refines if needed.
    
    Time complexity: O(n × bins^6) but adaptive
    Balances accuracy and performance.
    """
    # Start with coarse grid
    coverage_coarse = _calculate_coverage_grid_binning(
        coords, half_widths, bounds, coordinate_system, coarse_bins
    )
    
    # If coverage is low, use coarse estimate (sparse orbits, high accuracy)
    if coverage_coarse < coverage_threshold:
        return coverage_coarse
    
    # If coverage is high, refine the grid for better accuracy
    coverage_fine = _calculate_coverage_grid_binning(
        coords, half_widths, bounds, coordinate_system, fine_bins
    )
    
    return coverage_fine


def _calculate_coverage_grid_binning(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
    n_bins_per_dim: int
) -> float:
    """
    Grid binning coverage calculation - bins phase space and counts occupied bins.
    
    Time complexity: O(n × bins^6)
    No assumptions about orbit positions or overlaps.
    """
    import itertools
    
    coord_names = _get_coordinate_names(coordinate_system)
    ranges = [bounds[coord] for coord in coord_names]
    
    # Create bins for each dimension
    bin_edges = [np.linspace(r[0], r[1], n_bins_per_dim + 1) for r in ranges]
    
    # Track which bins are covered by any orbit volume
    covered_bins = set()
    
    for orbit_center in coords:
        # Find all bins this orbit volume touches
        orbit_min = orbit_center - half_widths
        orbit_max = orbit_center + half_widths
        
        # Get bin indices for this orbit's extent
        bin_indices = []
        for dim in range(len(coord_names)):
            min_bin = np.searchsorted(bin_edges[dim], orbit_min[dim], side='right') - 1
            max_bin = np.searchsorted(bin_edges[dim], orbit_max[dim], side='left')
            bin_indices.append(range(max(0, min_bin), min(n_bins_per_dim, max_bin + 1)))
        
        # Add all combinations of bin indices that this orbit covers
        for bin_combo in itertools.product(*bin_indices):
            covered_bins.add(bin_combo)
    
    # Coverage = fraction of bins covered
    total_bins = n_bins_per_dim ** len(coord_names)
    coverage_percentage = len(covered_bins) / total_bins * 100
    
    return coverage_percentage


def _calculate_coverage_exact(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
) -> float:
    """
    Exact coverage calculation using full overlap analysis.
    
    Time complexity: O(n²)
    Most accurate but slowest method.
    """
    n_orbits = len(coords)
    individual_volume = np.prod(2 * half_widths)
    
    # Calculate overlaps using existing exact analysis
    overlap_count, total_overlap_volume = _exact_overlap_analysis(coords, half_widths, n_orbits)
    
    # Calculate actual covered volume accounting for overlaps
    total_volume_with_overlap = n_orbits * individual_volume - total_overlap_volume
    
    # Get total phase space volume
    coord_names = _get_coordinate_names(coordinate_system)
    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    total_phase_space = np.prod(ranges)
    
    coverage_percentage = min(100.0, 100.0 * total_volume_with_overlap / total_phase_space)
    return coverage_percentage


def _get_coordinate_names(coordinate_system: str) -> List[str]:
    """Helper function to get coordinate names for a given system."""
    if coordinate_system == "spherical":
        return ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        return ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        return ["a", "e", "i", "raan", "ap", "M"]
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")



def _create_basic_report(
    coords: np.ndarray,
    half_widths: np.ndarray,
    bounds: Dict[str, Tuple[float, float]],
    coordinate_system: str,
) -> Dict:
    """
    Create a basic coverage report with only essential metrics.

    This provides the coverage_percentage needed by iterative functions without expensive O(n²) computations.
    """
    n_orbits = len(coords)
    individual_volume = np.prod(2 * half_widths)

    # Get the essential coverage percentage using fast method
    coverage_percentage = calculate_coverage_percentage(coords, half_widths, bounds, coordinate_system, method="fast")

    # Calculate total phase space volume for reporting
    if coordinate_system == "spherical":
        coord_names = ["rho", "lon", "lat", "vrho", "vlon", "vlat"]
    elif coordinate_system == "cartesian":
        coord_names = ["x", "y", "z", "vx", "vy", "vz"]
    elif coordinate_system == "keplerian":
        coord_names = ["a", "e", "i", "raan", "ap", "M"]

    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    total_phase_space = np.prod(ranges)
    total_volume_no_overlap = n_orbits * individual_volume

    # Basic volume statistics (no pairwise calculations)
    volume_stats = {
        "individual_volume": individual_volume,
        "total_volume_no_overlap": total_volume_no_overlap,
        "total_overlap_volume": 0.0,  # Not calculated
        "mean_center_distance": 0.0,  # Not calculated
    }

    return {
        "n_orbits": n_orbits,
        "coverage_percentage": coverage_percentage,
        "overlap_percentage": 0.0,  # Not calculated
        "n_overlapping_pairs": 0,  # Not calculated
        "efficiency": 1.0,  # Assume no overlap
        "volume_stats": volume_stats,
        "phase_space_volume": total_phase_space,
        "bounds": bounds,
        "half_widths": half_widths,
        "coordinate_system": coordinate_system,
        "analysis_skipped": True,  # Flag to indicate analysis was skipped
    }


def analyze_orbit_coverage_diagnostics(test_orbits: TestOrbits, orbit_volumes: OrbitVolumes) -> Dict:
    """
    Perform detailed coverage diagnostics on existing orbits and volumes.

    This is the expensive O(n²) analysis separated out for optional use.
    Takes TestOrbits and OrbitVolumes as input - works with any generation method.

    Parameters
    ----------
    test_orbits : TestOrbits
        Generated test orbits
    orbit_volumes : OrbitVolumes
        Volume information for each orbit

    Returns
    -------
    Dict
        Detailed diagnostics including overlap analysis, efficiency metrics, etc.
    """
    n_orbits = len(test_orbits)

    if n_orbits == 0:
        return {
            "n_orbits": 0,
            "overlap_percentage": 0.0,
            "n_overlapping_pairs": 0,
            "efficiency": 1.0,
            "mean_center_distance": 0.0,
            "diagnostics_note": "No orbits to analyze",
        }

    # Extract coordinates and half-widths from the orbit volumes
    # Get the first orbit's half-widths (assuming all orbits have same volume sizes)
    first_half_widths = np.array(list(orbit_volumes.half_widths[0].as_py()))
    coordinate_system = orbit_volumes.coordinate_system[0].as_py()

    # Build coords array
    coords = np.array([list(orbit_volumes.centers[i].as_py()) for i in range(n_orbits)])

    # Perform the expensive overlap analysis
    if n_orbits > 1000:
        overlap_count, total_overlap_volume = _fast_overlap_analysis(coords, first_half_widths, n_orbits)
    else:
        overlap_count, total_overlap_volume = _exact_overlap_analysis(coords, first_half_widths, n_orbits)

    # Calculate metrics
    individual_volume = np.prod(2 * first_half_widths)
    total_volume_no_overlap = n_orbits * individual_volume
    overlap_percentage = (
        100.0 * total_overlap_volume / total_volume_no_overlap if total_volume_no_overlap > 0 else 0.0
    )
    efficiency = (
        1.0 - (total_overlap_volume / total_volume_no_overlap) if total_volume_no_overlap > 0 else 1.0
    )

    # Calculate mean distance
    mean_distance = _calculate_mean_distance(coords) if n_orbits > 1 else 0.0

    return {
        "n_orbits": n_orbits,
        "overlap_percentage": overlap_percentage,
        "n_overlapping_pairs": overlap_count,
        "efficiency": efficiency,
        "mean_center_distance": mean_distance,
        "total_overlap_volume": total_overlap_volume,
        "individual_volume": individual_volume,
        "coordinate_system": coordinate_system,
        "diagnostics_note": f'Detailed analysis using {"statistical sampling" if n_orbits > 1000 else "exact calculation"}',
    }
