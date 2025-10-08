"""
Phase space coverage analysis for THOR test orbits.

This module provides a simple interface for generating test orbits with
even phase space coverage and analyzing their overlap characteristics.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
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
            "spherical": ['rho', 'lon', 'lat', 'vrho', 'vlon', 'vlat'],
            "cartesian": ['x', 'y', 'z', 'vx', 'vy', 'vz'],
            "keplerian": ['a', 'e', 'i', 'raan', 'ap', 'M']
        }
        
        if self.coordinate_system not in valid_systems:
            raise ValueError(f"coordinate_system must be one of {list(valid_systems.keys())}")
        
        expected_coords = valid_systems[self.coordinate_system]
        if set(self.bounds.keys()) != set(expected_coords):
            raise ValueError(f"For {self.coordinate_system} coordinates, bounds must contain: {expected_coords}")
    
    @classmethod
    def from_spherical(cls, rho: Tuple[float, float], lon: Tuple[float, float], 
                      lat: Tuple[float, float], vrho: Tuple[float, float],
                      vlon: Tuple[float, float], vlat: Tuple[float, float]) -> 'PhaseSpaceBounds':
        """Create bounds from spherical coordinate parameters."""
        return cls(
            coordinate_system="spherical",
            bounds={
                'rho': rho, 'lon': lon, 'lat': lat,
                'vrho': vrho, 'vlon': vlon, 'vlat': vlat
            }
        )
    
    @classmethod
    def from_cartesian(cls, x: Tuple[float, float], y: Tuple[float, float],
                      z: Tuple[float, float], vx: Tuple[float, float],
                      vy: Tuple[float, float], vz: Tuple[float, float]) -> 'PhaseSpaceBounds':
        """Create bounds from cartesian coordinate parameters."""
        return cls(
            coordinate_system="cartesian",
            bounds={
                'x': x, 'y': y, 'z': z,
                'vx': vx, 'vy': vy, 'vz': vz
            }
        )
    
    @classmethod
    def from_keplerian(cls, a: Tuple[float, float], e: Tuple[float, float],
                      i: Tuple[float, float], raan: Tuple[float, float],
                      ap: Tuple[float, float], M: Tuple[float, float]) -> 'PhaseSpaceBounds':
        """Create bounds from keplerian coordinate parameters."""
        return cls(
            coordinate_system="keplerian",
            bounds={
                'a': a, 'e': e, 'i': i,
                'raan': raan, 'ap': ap, 'M': M
            }
        )
    
    @classmethod
    def from_type(cls, asteroid_type: str, 
                 coordinate_system: str = "spherical") -> 'PhaseSpaceBounds':
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
            raise ValueError(f"Unknown asteroid type: {asteroid_type}. "
                           f"Available types: main_belt, near_earth, jupiter_trojans, comprehensive, inner_main_belt")
    
    @classmethod
    def _main_belt_bounds(cls, coordinate_system: str) -> 'PhaseSpaceBounds':
        """Main belt asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(1.8, 4.0), lon=(0.0, 360.0), lat=(-30.0, 30.0),
                vrho=(-0.01, 0.01), vlon=(-1.0, 1.0), vlat=(-1.0, 1.0)
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-4.0, 4.0), y=(-4.0, 4.0), z=(-1.0, 1.0),
                vx=(-0.02, 0.02), vy=(-0.02, 0.02), vz=(-0.01, 0.01)
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(1.8, 4.0), e=(0.0, 0.3), i=(0.0, 30.0),
                raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
    
    @classmethod
    def _near_earth_bounds(cls, coordinate_system: str) -> 'PhaseSpaceBounds':
        """Near-Earth asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(0.5, 3.0), lon=(0.0, 360.0), lat=(-45.0, 45.0),
                vrho=(-0.02, 0.02), vlon=(-2.0, 2.0), vlat=(-2.0, 2.0)
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-3.0, 3.0), y=(-3.0, 3.0), z=(-1.5, 1.5),
                vx=(-0.03, 0.03), vy=(-0.03, 0.03), vz=(-0.02, 0.02)
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(0.5, 3.0), e=(0.0, 0.8), i=(0.0, 45.0),
                raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
    
    @classmethod
    def _jupiter_trojans_bounds(cls, coordinate_system: str) -> 'PhaseSpaceBounds':
        """Jupiter Trojan asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(4.5, 5.8), lon=(0.0, 360.0), lat=(-25.0, 25.0),
                vrho=(-0.005, 0.005), vlon=(-0.5, 0.5), vlat=(-0.5, 0.5)
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-6.0, 6.0), y=(-6.0, 6.0), z=(-2.0, 2.0),
                vx=(-0.015, 0.015), vy=(-0.015, 0.015), vz=(-0.01, 0.01)
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(4.5, 5.8), e=(0.0, 0.2), i=(0.0, 25.0),
                raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
    
    @classmethod
    def _comprehensive_bounds(cls, coordinate_system: str) -> 'PhaseSpaceBounds':
        """Comprehensive bounds covering most asteroid populations."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(0.3, 5.5), lon=(0.0, 360.0), lat=(-90.0, 90.0),
                vrho=(-0.03, 0.03), vlon=(-3.0, 3.0), vlat=(-3.0, 3.0)
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-6.0, 6.0), y=(-6.0, 6.0), z=(-3.0, 3.0),
                vx=(-0.04, 0.04), vy=(-0.04, 0.04), vz=(-0.03, 0.03)
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(0.3, 5.5), e=(0.0, 0.9), i=(0.0, 90.0),
                raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
    
    @classmethod
    def _inner_main_belt_bounds(cls, coordinate_system: str) -> 'PhaseSpaceBounds':
        """Inner main belt asteroid bounds."""
        if coordinate_system == "spherical":
            return cls.from_spherical(
                rho=(1.8, 2.8), lon=(0.0, 360.0), lat=(-20.0, 20.0),
                vrho=(-0.008, 0.008), vlon=(-0.8, 0.8), vlat=(-0.8, 0.8)
            )
        elif coordinate_system == "cartesian":
            return cls.from_cartesian(
                x=(-3.0, 3.0), y=(-3.0, 3.0), z=(-0.8, 0.8),
                vx=(-0.015, 0.015), vy=(-0.015, 0.015), vz=(-0.008, 0.008)
            )
        elif coordinate_system == "keplerian":
            return cls.from_keplerian(
                a=(1.8, 2.8), e=(0.0, 0.25), i=(0.0, 20.0),
                raan=(0.0, 360.0), ap=(0.0, 360.0), M=(0.0, 360.0)
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
    centers = qv.LargeListColumn(pa.float64())      # 6D center coordinates
    half_widths = qv.LargeListColumn(pa.float64())  # 6D half-widths
    volume = qv.Float64Column()                     # Hypervolume
    
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
            if (center1[i] + half_width1[i] < center2[i] - half_width2[i] or
                center1[i] - half_width1[i] > center2[i] + half_width2[i]):
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
            min_bound = max(center1[i] - half_width1[i], 
                           center2[i] - half_width2[i])
            max_bound = min(center1[i] + half_width1[i], 
                           center2[i] + half_width2[i])
            overlap_widths[i] = max(0, max_bound - min_bound)
        
        return np.prod(overlap_widths)
    
    @classmethod
    def from_arrays(cls, orbit_ids: List[str], coordinate_system: str, 
                    centers: np.ndarray, half_widths: np.ndarray) -> 'OrbitVolumes':
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
            'rho': self.coord_0_center.to_numpy(zero_copy_only=False),
            'lon': self.coord_1_center.to_numpy(zero_copy_only=False),
            'lat': self.coord_2_center.to_numpy(zero_copy_only=False),
            'vrho': self.coord_3_center.to_numpy(zero_copy_only=False),
            'vlon': self.coord_4_center.to_numpy(zero_copy_only=False),
            'vlat': self.coord_5_center.to_numpy(zero_copy_only=False),
        }
    
    @property
    def cartesian_coords(self) -> dict:
        """Get cartesian coordinate values (only valid if coordinate_system == 'cartesian')."""
        if len(self) == 0 or self.coordinate_system[0].as_py() != "cartesian":
            raise ValueError("This OrbitVolumes is not in cartesian coordinates")
        
        return {
            'x': self.coord_0_center.to_numpy(zero_copy_only=False),
            'y': self.coord_1_center.to_numpy(zero_copy_only=False),
            'z': self.coord_2_center.to_numpy(zero_copy_only=False),
            'vx': self.coord_3_center.to_numpy(zero_copy_only=False),
            'vy': self.coord_4_center.to_numpy(zero_copy_only=False),
            'vz': self.coord_5_center.to_numpy(zero_copy_only=False),
        }
    
    @property
    def keplerian_coords(self) -> dict:
        """Get keplerian coordinate values (only valid if coordinate_system == 'keplerian')."""
        if len(self) == 0 or self.coordinate_system[0].as_py() != "keplerian":
            raise ValueError("This OrbitVolumes is not in keplerian coordinates")
        
        return {
            'a': self.coord_0_center.to_numpy(zero_copy_only=False),
            'e': self.coord_1_center.to_numpy(zero_copy_only=False),
            'i': self.coord_2_center.to_numpy(zero_copy_only=False),
            'raan': self.coord_3_center.to_numpy(zero_copy_only=False),
            'ap': self.coord_4_center.to_numpy(zero_copy_only=False),
            'M': self.coord_5_center.to_numpy(zero_copy_only=False),
        }


def generate_even_coverage_test_orbits(
    n_orbits: int,
    half_widths: Optional[np.ndarray] = None,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    no_analysis: bool = False,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate test orbits with even phase space coverage and analyze overlaps.
    
    This function creates test orbits distributed evenly across 6D orbital
    phase space and returns both the orbits and a coverage analysis report.
    
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
        Used only if bounds is None. Default: "main_belt"
    epoch : Timestamp, optional
        Epoch for the orbits. Default: J2000.0 
    frame : str, optional
        Coordinate frame. Default: "ecliptic"
    no_analysis : bool, optional
        If True, skip overlap analysis for faster orbit generation. Default: False
        
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
    logger.info(f"Generating {n_orbits} test orbits with even phase space coverage in {coordinate_system} coordinates")
    
    # Validate coordinate system
    if coordinate_system not in ["spherical", "cartesian", "keplerian"]:
        raise ValueError(f"coordinate_system must be 'spherical', 'cartesian', or 'keplerian', got '{coordinate_system}'")
    
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
            asteroid_type = "main_belt"
        
        bounds_obj = PhaseSpaceBounds.from_type(asteroid_type, coordinate_system)
        bounds = bounds_obj.bounds_dict
    elif hasattr(bounds, 'bounds_dict'):
        # It's a PhaseSpaceBounds object
        if bounds.coordinate_system != coordinate_system:
            raise ValueError(f"PhaseSpaceBounds coordinate system ({bounds.coordinate_system}) "
                           f"doesn't match requested coordinate_system ({coordinate_system})")
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
    centers_list = [coords_6d[i, :].tolist() for i in range(n_points)]
    half_widths_list = [half_widths.tolist() for _ in range(n_points)]
    
    orbit_volumes_data = {
        "orbit_id": orbit_ids,
        "coordinate_system": [coordinate_system] * n_points,
        "centers": centers_list,
        "half_widths": half_widths_list,
        "volume": np.full(n_points, individual_volume),
    }
    
    orbit_volumes = OrbitVolumes.from_kwargs(**orbit_volumes_data)
    
    # Analyze coverage (optional)
    if no_analysis:
        # Skip expensive overlap analysis - just return basic info
        report = _create_basic_report(coords_6d, half_widths, bounds, coordinate_system)
        logger.info(f"Generated {len(test_orbits)} test orbits (analysis skipped)")
    else:
        # Full analysis including overlap checking
        report = _analyze_coverage(coords_6d, half_widths, bounds, coordinate_system)
        logger.info(f"Generated {len(test_orbits)} test orbits")
        logger.info(f"Coverage: {report['coverage_percentage']:.1f}%, "
                    f"Overlap: {report['overlap_percentage']:.1f}%, "
                    f"Efficiency: {report['efficiency']:.3f}")
    
    return test_orbits, orbit_volumes, report


def _generate_even_grid_points(n_orbits: int, bounds: Dict[str, Tuple[float, float]], coordinate_system: str) -> np.ndarray:
    """
    Generate evenly distributed points in 6D phase space using a grid approach.
    
    Parameters
    ----------
    n_orbits : int
        Target number of orbits
    bounds : Dict[str, Tuple[float, float]]
        Phase space bounds
        
    Returns
    -------
    np.ndarray
        Array of shape (n_actual, 6) with coordinates
    """
    # Get coordinate names based on system
    if coordinate_system == "spherical":
        coord_names = ['rho', 'lon', 'lat', 'vrho', 'vlon', 'vlat']
    elif coordinate_system == "cartesian":
        coord_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    elif coordinate_system == "keplerian":
        coord_names = ['a', 'e', 'i', 'raan', 'ap', 'M']
    
    # Calculate grid dimensions to get as close as possible to n_orbits
    # Start with cube root as base, then iteratively adjust
    base_n = max(1, int(np.ceil(n_orbits**(1/6))))
    
    # Get coordinate ranges for proportional scaling
    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    range_weights = ranges / np.mean(ranges)  # Normalize to mean=1
    
    # Try different scaling approaches to find best grid
    best_grid = None
    best_error = float('inf')
    
    # Try several different approaches
    for approach in range(5):
        if approach == 0:
            # Uniform grid
            grid_dims = np.full(6, max(1, int(np.ceil(n_orbits**(1/6)))))
        elif approach == 1:
            # Proportional to ranges
            grid_dims = np.maximum(1, (base_n * range_weights**(1/3)).astype(int))
        elif approach == 2:
            # Slightly larger base
            larger_base = max(1, int(np.ceil((n_orbits * 1.1)**(1/6))))
            grid_dims = np.maximum(1, (larger_base * range_weights**(1/4)).astype(int))
        elif approach == 3:
            # Focus more points on larger dimensions
            grid_dims = np.maximum(1, (base_n * range_weights**(1/2)).astype(int))
        else:
            # Try to balance by adjusting individual dimensions
            grid_dims = np.maximum(1, (base_n * range_weights**(1/5)).astype(int))
        
        # Fine-tune to get closer to target
        total_points = np.prod(grid_dims)
        if total_points < n_orbits:
            # Need more points - try increasing dimensions iteratively
            for _ in range(10):  # Max 10 adjustments
                # Find dimension that gives best improvement
                best_dim = -1
                best_improvement = 0
                for dim in range(6):
                    test_dims = grid_dims.copy()
                    test_dims[dim] += 1
                    new_total = np.prod(test_dims)
                    if new_total <= n_orbits * 1.1:  # Don't go too far over
                        improvement = new_total - total_points
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_dim = dim
                
                if best_dim >= 0:
                    grid_dims[best_dim] += 1
                    total_points = np.prod(grid_dims)
                else:
                    break
        
        # Calculate error from target
        error = abs(total_points - n_orbits)
        if error < best_error:
            best_error = error
            best_grid = grid_dims.copy()
        
        # If we found exact match, use it
        if total_points == n_orbits:
            best_grid = grid_dims.copy()
            break
    
    grid_dims = best_grid
    total_points = np.prod(grid_dims)
    
    logger.info(f"Using grid dimensions: {grid_dims} (total: {total_points} points)")
    
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
    mesh = np.meshgrid(*grid_coords, indexing='ij')
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


def _create_coordinates(coords_6d: np.ndarray, coordinate_system: str, 
                       time: Timestamp, origin: Origin, frame: str):
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


def _analyze_coverage(coords: np.ndarray, half_widths: np.ndarray, 
                     bounds: Dict[str, Tuple[float, float]], coordinate_system: str) -> Dict:
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
        coord_names = ['rho', 'lon', 'lat', 'vrho', 'vlon', 'vlat']
    elif coordinate_system == "cartesian":
        coord_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    elif coordinate_system == "keplerian":
        coord_names = ['a', 'e', 'i', 'raan', 'ap', 'M']
    
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
    overlap_percentage = 100.0 * total_overlap_volume / total_volume_no_overlap if total_volume_no_overlap > 0 else 0.0
    efficiency = 1.0 - (total_overlap_volume / total_volume_no_overlap) if total_volume_no_overlap > 0 else 1.0
    
    # Volume statistics
    volume_stats = {
        'individual_volume': individual_volume,
        'total_volume_no_overlap': total_volume_no_overlap,
        'total_overlap_volume': total_overlap_volume,
        'mean_center_distance': _calculate_mean_distance(coords) if n_orbits > 1 else 0.0,
    }
    
    return {
        'n_orbits': n_orbits,
        'coverage_percentage': coverage_percentage,
        'overlap_percentage': overlap_percentage,
        'n_overlapping_pairs': overlap_count,
        'efficiency': efficiency,
        'volume_stats': volume_stats,
        'phase_space_volume': total_phase_space,
        'bounds': bounds,
        'half_widths': half_widths.tolist(),
        'coordinate_system': coordinate_system,
    }


def generate_orbit_volumes_for_target_coverage(
    n_orbits: int,
    target_coverage_percent: float,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    max_iterations: int = 10,
    tolerance: float = 2.0,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate test orbits with volumes sized to achieve a target coverage percentage.
    
    This function iteratively adjusts volume sizes to reach the desired phase space
    coverage with the specified number of orbits.
    
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
    max_iterations : int, optional
        Maximum number of iterations to converge on target coverage. Default: 10
    tolerance : float, optional
        Acceptable tolerance in coverage percentage. Default: 2.0%
        
    Returns
    -------
    Tuple[TestOrbits, OrbitVolumes, Dict]
        - TestOrbits: Generated test orbits
        - OrbitVolumes: Volume information for each orbit (quivr Table)
        - Dict: Coverage analysis report with additional keys:
            - 'target_coverage_percent': Requested coverage percentage
            - 'iterations_used': Number of iterations needed
            - 'converged': Whether the algorithm converged within tolerance
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
    
    # Calculate phase space volume
    phase_space_volume = bounds.volume
    
    # Initial estimate: assume minimal overlap for target coverage
    # Target volume = (target_coverage / 100) * phase_space_volume
    target_total_volume = (target_coverage_percent / 100.0) * phase_space_volume
    
    # Estimate individual volume size (assuming no overlap initially)
    target_individual_volume = target_total_volume / n_orbits
    
    # Convert to half-widths (assuming uniform scaling across dimensions)
    # For 6D hyperrectangle: volume = prod(2 * half_widths)
    coord_ranges = bounds.ranges
    relative_ranges = coord_ranges / np.sum(coord_ranges)  # Normalize
    
    # Scale factor to achieve target volume
    scale_factor = (target_individual_volume / np.prod(coord_ranges)) ** (1/6)
    initial_half_widths = scale_factor * coord_ranges / 2
    
    logger.info(f"Initial volume estimate: {target_individual_volume:.2e}")
    logger.info(f"Initial half-widths scale factor: {scale_factor:.4f}")
    
    # Iterative refinement
    current_half_widths = initial_half_widths.copy()
    best_result = None
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Generate orbits with current half-widths
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
            n_orbits=n_orbits,
            half_widths=current_half_widths,
            bounds=bounds,
            coordinate_system=coordinate_system,
            epoch=epoch,
            frame=frame,
        )
        
        current_coverage = report['coverage_percentage']
        error = abs(current_coverage - target_coverage_percent)
        
        logger.info(f"  Current coverage: {current_coverage:.1f}% (target: {target_coverage_percent:.1f}%)")
        logger.info(f"  Error: {error:.1f}%")
        
        # Track best result
        if error < best_error:
            best_error = error
            best_result = (test_orbits, orbit_volumes, report)
        
        # Check convergence
        if error <= tolerance:
            logger.info(f"Converged in {iteration + 1} iterations")
            report['target_coverage_percent'] = target_coverage_percent
            report['iterations_used'] = iteration + 1
            report['converged'] = True
            return test_orbits, orbit_volumes, report
        
        # Adjust half-widths for next iteration
        if iteration < max_iterations - 1:  # Don't adjust on last iteration
            # Simple scaling approach: if coverage is too low, increase volumes
            coverage_ratio = target_coverage_percent / max(current_coverage, 0.1)  # Avoid division by zero
            
            # Apply cube root since we're scaling 6D volumes
            scale_adjustment = coverage_ratio ** (1/6)
            
            # Limit adjustment to prevent oscillation
            scale_adjustment = np.clip(scale_adjustment, 0.5, 2.0)
            
            current_half_widths *= scale_adjustment
            
            logger.info(f"  Adjusting volumes by factor {scale_adjustment:.3f}")
    
    # If we didn't converge, return the best result
    logger.warning(f"Did not converge within {max_iterations} iterations. Best error: {best_error:.1f}%")
    
    if best_result is not None:
        test_orbits, orbit_volumes, report = best_result
        report['target_coverage_percent'] = target_coverage_percent
        report['iterations_used'] = max_iterations
        report['converged'] = False
        return test_orbits, orbit_volumes, report
    else:
        raise RuntimeError("Failed to generate any valid orbits")


def generate_orbits_for_coverage_with_fixed_volumes(
    target_coverage_percent: float,
    half_widths: np.ndarray,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], PhaseSpaceBounds]] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
    max_orbits: int = 1000,
    tolerance: float = 2.0,
) -> Tuple[TestOrbits, OrbitVolumes, Dict]:
    """
    Generate as many test orbits as needed to achieve target coverage with fixed volume sizes.
    
    This function iteratively increases the number of orbits until the desired phase space
    coverage is achieved with the specified volume sizes.
    
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
        Maximum number of orbits to generate. Default: 1000
    tolerance : float, optional
        Acceptable tolerance in coverage percentage. Default: 2.0%
        
    Returns
    -------
    Tuple[TestOrbits, OrbitVolumes, Dict]
        - TestOrbits: Generated test orbits
        - OrbitVolumes: Volume information for each orbit (quivr Table)
        - Dict: Coverage analysis report with additional keys:
            - 'target_coverage_percent': Requested coverage percentage
            - 'attempts': List of (n_orbits, coverage) attempts made
            - 'converged': Whether the algorithm converged within tolerance
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
    
    # Calculate individual volume and estimate initial number of orbits needed
    individual_volume = np.prod(2 * half_widths)
    phase_space_volume = bounds.volume
    target_total_volume = (target_coverage_percent / 100.0) * phase_space_volume
    
    # Initial estimate (assuming no overlap)
    initial_estimate = max(1, int(np.ceil(target_total_volume / individual_volume)))
    
    logger.info(f"Individual volume: {individual_volume:.2e}")
    logger.info(f"Phase space volume: {phase_space_volume:.2e}")
    logger.info(f"Initial orbit estimate: {initial_estimate}")
    
    # Track attempts and results
    attempts = []
    best_result = None
    best_error = float('inf')
    
    # Start with initial estimate and adjust
    current_n_orbits = initial_estimate
    
    # Search strategy: start with estimate, then increase if coverage is too low
    search_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    
    for multiplier in search_multipliers:
        n_orbits_to_try = max(1, int(current_n_orbits * multiplier))
        
        # Don't exceed max_orbits
        if n_orbits_to_try > max_orbits:
            n_orbits_to_try = max_orbits
        
        logger.info(f"Trying {n_orbits_to_try} orbits (multiplier: {multiplier:.1f})")
        
        # Generate orbits with current count
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
            n_orbits=n_orbits_to_try,
            half_widths=half_widths,
            bounds=bounds,
            coordinate_system=coordinate_system,
            epoch=epoch,
            frame=frame,
        )
        
        current_coverage = report['coverage_percentage']
        error = abs(current_coverage - target_coverage_percent)
        
        logger.info(f"  Generated {len(test_orbits)} orbits")
        logger.info(f"  Coverage: {current_coverage:.1f}% (target: {target_coverage_percent:.1f}%)")
        logger.info(f"  Error: {error:.1f}%")
        
        # Track this attempt
        attempts.append((len(test_orbits), current_coverage))
        
        # Track best result
        if error < best_error:
            best_error = error
            best_result = (test_orbits, orbit_volumes, report)
        
        # Check if we've achieved the target
        if error <= tolerance:
            logger.info(f"Achieved target coverage with {len(test_orbits)} orbits")
            report['target_coverage_percent'] = target_coverage_percent
            report['attempts'] = attempts
            report['converged'] = True
            return test_orbits, orbit_volumes, report
        
        # If coverage is still too low and we haven't hit max_orbits, continue
        if current_coverage < target_coverage_percent and n_orbits_to_try < max_orbits:
            continue
        else:
            # Either we overshot the target or hit max_orbits
            break
    
    # If we didn't converge, return the best result
    logger.warning(f"Did not converge within tolerance. Best error: {best_error:.1f}%")
    
    if best_result is not None:
        test_orbits, orbit_volumes, report = best_result
        report['target_coverage_percent'] = target_coverage_percent
        report['attempts'] = attempts
        report['converged'] = False
        return test_orbits, orbit_volumes, report
    else:
        raise RuntimeError("Failed to generate any valid orbits")


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


def _create_basic_report(coords: np.ndarray, half_widths: np.ndarray, 
                        bounds: Dict[str, Tuple[float, float]], coordinate_system: str) -> Dict:
    """
    Create a basic coverage report without expensive overlap analysis.
    
    This provides essential metrics without O(n²) computations.
    """
    n_orbits = len(coords)
    individual_volume = np.prod(2 * half_widths)
    
    # Get coordinate names based on system
    if coordinate_system == "spherical":
        coord_names = ['rho', 'lon', 'lat', 'vrho', 'vlon', 'vlat']
    elif coordinate_system == "cartesian":
        coord_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    elif coordinate_system == "keplerian":
        coord_names = ['a', 'e', 'i', 'raan', 'ap', 'M']
    
    # Calculate total phase space volume
    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    total_phase_space = np.prod(ranges)
    
    # Basic coverage estimate (no overlap analysis)
    total_volume_no_overlap = n_orbits * individual_volume
    coverage_percentage = min(100.0, 100.0 * total_volume_no_overlap / total_phase_space)
    
    # Basic volume statistics (no pairwise calculations)
    volume_stats = {
        'individual_volume': individual_volume,
        'total_volume_no_overlap': total_volume_no_overlap,
        'total_overlap_volume': 0.0,  # Not calculated
        'mean_center_distance': 0.0,  # Not calculated
    }
    
    return {
        'n_orbits': n_orbits,
        'coverage_percentage': coverage_percentage,
        'overlap_percentage': 0.0,  # Not calculated
        'n_overlapping_pairs': 0,   # Not calculated
        'efficiency': 1.0,          # Assume no overlap
        'volume_stats': volume_stats,
        'phase_space_volume': total_phase_space,
        'bounds': bounds,
        'half_widths': half_widths.tolist(),
        'coordinate_system': coordinate_system,
        'analysis_skipped': True,   # Flag to indicate analysis was skipped
    }