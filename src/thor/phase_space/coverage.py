"""
Phase space coverage analysis for THOR test orbits.

This module provides a simple interface for generating test orbits with
even phase space coverage and analyzing their overlap characteristics.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.origin import Origin, OriginCodes
from adam_core.time import Timestamp

from ..orbit import TestOrbits

logger = logging.getLogger(__name__)


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
    bounds: Optional[Union[Dict[str, Tuple[float, float]], 'PhaseSpaceBounds']] = None,
    coordinate_system: str = "spherical",
    asteroid_type: Optional[str] = None,
    epoch: Optional[Timestamp] = None,
    frame: str = "ecliptic",
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
        # Import here to avoid circular imports
        from .bounds import PhaseSpaceBounds
        
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
    
    # Analyze coverage
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
    # Calculate grid dimensions for approximately n_orbits points
    # Use cube root as starting point, then adjust per dimension
    base_n = int(np.ceil(n_orbits**(1/6)))
    
    # Get coordinate names based on system
    if coordinate_system == "spherical":
        coord_names = ['rho', 'lon', 'lat', 'vrho', 'vlon', 'vlat']
    elif coordinate_system == "cartesian":
        coord_names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    elif coordinate_system == "keplerian":
        coord_names = ['a', 'e', 'i', 'raan', 'ap', 'M']
    
    # Adjust grid size per dimension based on range size
    ranges = np.array([bounds[coord][1] - bounds[coord][0] for coord in coord_names])
    
    # Scale grid dimensions proportionally to ranges
    range_weights = ranges / np.mean(ranges)
    grid_dims = np.maximum(1, (base_n * range_weights**(1/3)).astype(int))
    
    # Adjust to get closer to target
    total_points = np.prod(grid_dims)
    if total_points < n_orbits * 0.7:
        # Too few points, increase dimensions
        scale_factor = (n_orbits / total_points)**(1/6)
        grid_dims = np.maximum(1, (grid_dims * scale_factor).astype(int))
    elif total_points > n_orbits * 1.5:
        # Too many points, decrease dimensions  
        scale_factor = (n_orbits / total_points)**(1/6)
        grid_dims = np.maximum(1, (grid_dims * scale_factor).astype(int))
    
    logger.info(f"Using grid dimensions: {grid_dims} (total: {np.prod(grid_dims)} points)")
    
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
    
    # If we have too many points, randomly sample to get closer to target
    if len(points) > n_orbits * 1.2:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(points), size=n_orbits, replace=False)
        points = points[indices]
    
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
    
    # Check pairwise overlaps
    for i in range(n_orbits):
        for j in range(i + 1, n_orbits):
            # Check if volumes overlap (simplified rectangular check)
            overlaps = True
            overlap_dims = []
            
            for dim in range(6):
                center_i, center_j = coords[i, dim], coords[j, dim]
                half_width = half_widths[dim]
                
                # Check overlap in this dimension
                if abs(center_i - center_j) > 2 * half_width:
                    overlaps = False
                    break
                else:
                    # Calculate overlap width in this dimension
                    overlap_width = 2 * half_width - abs(center_i - center_j)
                    overlap_dims.append(overlap_width)
            
            if overlaps:
                overlap_count += 1
                # Approximate overlap volume
                overlap_vol = np.prod(overlap_dims)
                total_overlap_volume += overlap_vol
    
    # Calculate metrics
    overlap_percentage = 100.0 * total_overlap_volume / total_volume_no_overlap if total_volume_no_overlap > 0 else 0.0
    efficiency = 1.0 - (total_overlap_volume / total_volume_no_overlap) if total_volume_no_overlap > 0 else 1.0
    
    # Volume statistics
    volume_stats = {
        'individual_volume': individual_volume,
        'total_volume_no_overlap': total_volume_no_overlap,
        'total_overlap_volume': total_overlap_volume,
        'mean_center_distance': np.mean([
            np.linalg.norm(coords[i] - coords[j]) 
            for i in range(n_orbits) for j in range(i + 1, n_orbits)
        ]) if n_orbits > 1 else 0.0,
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


def generate_orbits_for_target_coverage(
    n_orbits: int,
    target_coverage_percent: float,
    bounds: Optional[Union[Dict[str, Tuple[float, float]], 'PhaseSpaceBounds']] = None,
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
    from .bounds import PhaseSpaceBounds
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
    bounds: Optional[Union[Dict[str, Tuple[float, float]], 'PhaseSpaceBounds']] = None,
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
    from .bounds import PhaseSpaceBounds
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