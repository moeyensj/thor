"""
Phase space bounds definitions for different asteroid populations.

This module defines the bounds of "interesting" regions in 6D orbital
phase space for various asteroid populations in different coordinate systems.
"""

from typing import Dict, Tuple, Optional, Literal
import numpy as np
from dataclasses import dataclass


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


 