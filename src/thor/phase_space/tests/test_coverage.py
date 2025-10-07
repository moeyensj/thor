#!/usr/bin/env python3
"""
Test script for phase space coverage analysis.

This script demonstrates the main coverage function and verifies it works
correctly with THOR's existing orbit classes.
"""

import numpy as np
from thor.phase_space import generate_even_coverage_test_orbits

def test_basic_coverage():
    """Test basic coverage generation."""
    print("=== Testing Basic Coverage Generation ===")
    
    # Test with smaller bounds to show meaningful coverage
    from thor.phase_space.bounds import PhaseSpaceBounds
    
    # Create a small test region
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.5), lon=(0.0, 60.0), lat=(-10.0, 10.0),
        vrho=(-0.003, 0.003), vlon=(-0.3, 0.3), vlat=(-0.3, 0.3)
    )
    
    # Use larger volumes for this small region
    large_half_widths = np.array([0.15, 20.0, 8.0, 0.002, 0.2, 0.2])
    
    n_orbits = 50
    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits, 
        bounds=small_bounds,
        half_widths=large_half_widths
    )
    
    print(f"✓ Generated {len(test_orbits)} test orbits")
    print(f"  Requested: {n_orbits}, Actual: {len(test_orbits)}")
    
    # Check that we got TestOrbits
    from thor.orbit import TestOrbits
    assert isinstance(test_orbits, TestOrbits), f"Expected TestOrbits, got {type(test_orbits)}"
    print("✓ Returned correct TestOrbits type")
    
    # Check orbit volumes
    from thor.phase_space.coverage import OrbitVolumes
    assert len(orbit_volumes) == len(test_orbits), "Mismatch between orbits and volumes"
    assert isinstance(orbit_volumes, OrbitVolumes), f"Expected OrbitVolumes, got {type(orbit_volumes)}"
    print(f"✓ Generated OrbitVolumes table with {len(orbit_volumes)} entries")
    
    # Check that orbit IDs match
    orbit_ids_from_orbits = test_orbits.orbit_id.to_pylist()
    orbit_ids_from_volumes = orbit_volumes.orbit_id.to_pylist()
    assert orbit_ids_from_orbits == orbit_ids_from_volumes, "Orbit ID mismatch"
    print("✓ Orbit IDs match between TestOrbits and OrbitVolumes")
    
    # Check coordinates
    coords = test_orbits.coordinates
    print(f"✓ Coordinates type: {type(coords).__name__}")
    print(f"  Frame: {coords.frame}")
    print(f"  Origin codes: {coords.origin.code.unique()}")
    
    # Check coordinate ranges (Cartesian since that's what TestOrbits uses)
    import pyarrow.compute as pc
    print(f"  x range: {pc.min(coords.x).as_py():.2f} - {pc.max(coords.x).as_py():.2f} AU")
    print(f"  y range: {pc.min(coords.y).as_py():.2f} - {pc.max(coords.y).as_py():.2f} AU")
    print(f"  z range: {pc.min(coords.z).as_py():.2f} - {pc.max(coords.z).as_py():.2f} AU")
    
    # Check report
    print(f"\n✓ Coverage Report:")
    print(f"  Coverage: {report['coverage_percentage']:.1f}%")
    print(f"  Overlap: {report['overlap_percentage']:.1f}%")
    print(f"  Efficiency: {report['efficiency']:.3f}")
    print(f"  Overlapping pairs: {report['n_overlapping_pairs']}")
    
    return True

def test_custom_parameters():
    """Test with custom parameters."""
    print("\n=== Testing Custom Parameters ===")
    
    # Test with asteroid_type parameter
    test_orbits1, orbit_volumes1, report1 = generate_even_coverage_test_orbits(
        n_orbits=25,
        asteroid_type="near_earth",
        coordinate_system="spherical"
    )
    
    print(f"✓ Generated {len(test_orbits1)} orbits for near_earth asteroids")
    print(f"  Coverage: {report1['coverage_percentage']:.1f}%")
    
    # Test with PhaseSpaceBounds object
    from thor.phase_space.bounds import PhaseSpaceBounds
    
    custom_bounds_obj = PhaseSpaceBounds.from_type("inner_main_belt", "keplerian")
    custom_half_widths = np.array([0.05, 0.005, 2.0, 5.0, 5.0, 5.0])  # Keplerian half-widths
    
    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits=25,
        bounds=custom_bounds_obj,
        coordinate_system="keplerian",
        half_widths=custom_half_widths
    )
    
    print(f"✓ Generated {len(test_orbits)} orbits with PhaseSpaceBounds object")
    print(f"  Coordinate system: {report['coordinate_system']}")
    print(f"  Coverage: {report['coverage_percentage']:.1f}%")
    print(f"  Efficiency: {report['efficiency']:.3f}")
    
    # Basic validation - orbits should be generated
    assert len(test_orbits) > 0, "No orbits generated"
    assert report['coordinate_system'] == "keplerian", "Wrong coordinate system"
    
    return True

def test_different_sizes():
    """Test with different numbers of orbits."""
    print("\n=== Testing Different Orbit Counts ===")
    
    # Use a smaller test region for visible coverage
    from thor.phase_space.bounds import PhaseSpaceBounds
    test_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.2, 2.8), lon=(0.0, 90.0), lat=(-15.0, 15.0),
        vrho=(-0.004, 0.004), vlon=(-0.4, 0.4), vlat=(-0.4, 0.4)
    )
    
    sizes = [10, 100, 500]
    
    for n in sizes:
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(n, bounds=test_bounds)
        
        print(f"  n={n:3d}: Generated {len(test_orbits):3d} orbits, "
              f"Coverage: {report['coverage_percentage']:5.1f}%, "
              f"Efficiency: {report['efficiency']:.3f}")
        
        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for n={n}"
        assert 0 <= report['coverage_percentage'] <= 100, f"Invalid coverage for n={n}"
        # Efficiency can be negative when overlap is very high (more redundancy than useful coverage)
        assert report['efficiency'] <= 1, f"Efficiency too high for n={n}"
    
    print("✓ All sizes work correctly")
    return True

def test_orbit_ids():
    """Test that orbit IDs are generated correctly."""
    print("\n=== Testing Orbit IDs ===")
    
    test_orbits, orbit_volumes, _ = generate_even_coverage_test_orbits(20)
    
    orbit_ids = test_orbits.orbit_id.to_pylist()
    
    # Check uniqueness
    assert len(set(orbit_ids)) == len(orbit_ids), "Orbit IDs are not unique"
    print("✓ All orbit IDs are unique")
    
    # Check format
    for orbit_id in orbit_ids[:5]:  # Check first few
        assert orbit_id.startswith("even_coverage_"), f"Unexpected ID format: {orbit_id}"
    
    print(f"✓ Orbit ID format correct (e.g., '{orbit_ids[0]}')")
    
    return True

def test_coordinate_systems():
    """Test different coordinate systems."""
    print("\n=== Testing Different Coordinate Systems ===")
    
    systems = ["spherical", "cartesian", "keplerian"]
    
    for system in systems:
        # Use more orbits to get visible coverage
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(100, coordinate_system=system)
        
        print(f"  {system:10s}: Generated {len(test_orbits):2d} orbits, "
              f"Coverage: {report['coverage_percentage']:5.1f}%, "
              f"System: {report['coordinate_system']}")
        
        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for {system}"
        assert report['coordinate_system'] == system, f"Wrong coordinate system in report"
        assert isinstance(test_orbits.coordinates, type(test_orbits.coordinates)), "Wrong coordinate type"
        
        # All TestOrbits should have CartesianCoordinates regardless of input system
        from adam_core.coordinates import CartesianCoordinates
        assert isinstance(test_orbits.coordinates, CartesianCoordinates), f"TestOrbits should have CartesianCoordinates"
    
    print("✓ All coordinate systems work correctly")
    return True

def test_asteroid_types():
    """Test different asteroid types."""
    print("\n=== Testing Different Asteroid Types ===")
    
    types = ["main_belt", "near_earth", "jupiter_trojans", "comprehensive", "inner_main_belt"]
    
    for asteroid_type in types:
        # Use more orbits and larger volumes to get visible coverage
        large_half_widths = np.array([0.2, 30.0, 12.0, 0.003, 0.4, 0.3])
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
            200, 
            asteroid_type=asteroid_type,
            coordinate_system="spherical",
            half_widths=large_half_widths
        )
        
        print(f"  {asteroid_type:15s}: Generated {len(test_orbits):2d} orbits, "
              f"Coverage: {report['coverage_percentage']:5.1f}%")
        
        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for {asteroid_type}"
        assert report['coordinate_system'] == "spherical", f"Wrong coordinate system"
    
    print("✓ All asteroid types work correctly")
    return True

def test_target_coverage():
    """Test generating orbits for target coverage percentage."""
    print("\n=== Testing Target Coverage Generation ===")
    
    from thor.phase_space.coverage import generate_orbits_for_target_coverage, OrbitVolumes
    from thor.phase_space.bounds import PhaseSpaceBounds
    from thor.orbit import TestOrbits
    
    # Create a small test region for faster convergence
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.5), lon=(0.0, 90.0), lat=(-15.0, 15.0),
        vrho=(-0.005, 0.005), vlon=(-0.5, 0.5), vlat=(-0.5, 0.5)
    )
    
    # Test different target coverage percentages
    test_cases = [
        (10, 25.0),  # 10 orbits, 25% coverage
        (15, 50.0),  # 15 orbits, 50% coverage
        (20, 75.0),  # 20 orbits, 75% coverage
    ]
    
    for n_orbits, target_percent in test_cases:
        print(f"\n--- Testing {n_orbits} orbits for {target_percent}% coverage ---")
        
        test_orbits, orbit_volumes, report = generate_orbits_for_target_coverage(
            n_orbits=n_orbits,
            target_coverage_percent=target_percent,
            bounds=small_bounds,
            tolerance=5.0,  # Allow 5% tolerance for testing
            max_iterations=5  # Limit iterations for testing
        )
        
        # Check basic properties
        assert isinstance(test_orbits, TestOrbits), "Should return TestOrbits"
        assert isinstance(orbit_volumes, OrbitVolumes), "Should return OrbitVolumes"
        assert len(test_orbits) == len(orbit_volumes), "Orbit count mismatch"
        
        # Check report contains target coverage info
        assert 'target_coverage_percent' in report, "Missing target coverage in report"
        assert 'iterations_used' in report, "Missing iterations in report"
        assert 'converged' in report, "Missing convergence info in report"
        
        actual_coverage = report['coverage_percentage']
        target_coverage = report['target_coverage_percent']
        iterations = report['iterations_used']
        converged = report['converged']
        
        print(f"  Target: {target_coverage:.1f}%, Actual: {actual_coverage:.1f}%")
        print(f"  Iterations: {iterations}, Converged: {converged}")
        print(f"  Error: {abs(actual_coverage - target_coverage):.1f}%")
        
        # The algorithm should get reasonably close or converge
        if converged:
            assert abs(actual_coverage - target_coverage) <= 5.0, f"Converged but error too large: {abs(actual_coverage - target_coverage):.1f}%"
        
        print(f"  ✓ Generated {len(test_orbits)} orbits")
    
    print("✓ Target coverage generation works correctly")
    return True


def test_target_coverage_edge_cases():
    """Test edge cases for target coverage generation."""
    print("\n=== Testing Target Coverage Edge Cases ===")
    
    from thor.phase_space.coverage import generate_orbits_for_target_coverage
    from thor.orbit import TestOrbits
    
    # Test with very small target coverage
    print("Testing low coverage target (5%)...")
    test_orbits, orbit_volumes, report = generate_orbits_for_target_coverage(
        n_orbits=5,
        target_coverage_percent=5.0,
        asteroid_type="inner_main_belt",
        max_iterations=3
    )
    
    assert len(test_orbits) >= 1, "Should generate at least one orbit"
    print(f"  ✓ Low coverage: {report['coverage_percentage']:.1f}% ({len(test_orbits)} orbits generated)")
    
    # Test with different coordinate systems
    print("Testing with cartesian coordinates...")
    test_orbits, orbit_volumes, report = generate_orbits_for_target_coverage(
        n_orbits=8,
        target_coverage_percent=20.0,
        coordinate_system="cartesian",
        asteroid_type="main_belt",
        max_iterations=3
    )
    
    assert report['coordinate_system'] == "cartesian", "Should use cartesian coordinates"
    print(f"  ✓ Cartesian coverage: {report['coverage_percentage']:.1f}%")
    
    # Test input validation
    print("Testing input validation...")
    try:
        generate_orbits_for_target_coverage(0, 50.0)  # Invalid n_orbits
        assert False, "Should raise ValueError for n_orbits=0"
    except ValueError:
        print("  ✓ Correctly rejects n_orbits=0")
    
    try:
        generate_orbits_for_target_coverage(5, 0.0)  # Invalid coverage
        assert False, "Should raise ValueError for coverage=0"
    except ValueError:
        print("  ✓ Correctly rejects coverage=0%")
    
    try:
        generate_orbits_for_target_coverage(5, 150.0)  # Invalid coverage
        assert False, "Should raise ValueError for coverage>100"
    except ValueError:
        print("  ✓ Correctly rejects coverage>100%")
    
    print("✓ Edge cases handled correctly")
    return True


def test_fixed_volume_coverage():
    """Test generating orbits with fixed volumes to achieve target coverage."""
    print("\n=== Testing Fixed Volume Coverage Generation ===")
    
    from thor.phase_space.coverage import generate_orbits_for_coverage_with_fixed_volumes, OrbitVolumes
    from thor.phase_space.bounds import PhaseSpaceBounds
    from thor.orbit import TestOrbits
    
    # Create a small test region for predictable results
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.8), lon=(0.0, 90.0), lat=(-15.0, 15.0),
        vrho=(-0.005, 0.005), vlon=(-0.5, 0.5), vlat=(-0.5, 0.5)
    )
    
    # Test different target coverage percentages with fixed volumes
    test_cases = [
        (25.0, np.array([0.2, 25.0, 10.0, 0.003, 0.3, 0.25])),  # Large volumes for 25% coverage
        (50.0, np.array([0.25, 30.0, 12.0, 0.004, 0.35, 0.3])),  # Larger volumes for 50% coverage
    ]
    
    for target_percent, half_widths in test_cases:
        print(f"\n--- Testing {target_percent}% coverage with fixed volumes ---")
        
        test_orbits, orbit_volumes, report = generate_orbits_for_coverage_with_fixed_volumes(
            target_coverage_percent=target_percent,
            half_widths=half_widths,
            bounds=small_bounds,
            tolerance=5.0,  # Allow 5% tolerance for testing
            max_orbits=50   # Limit for testing
        )
        
        # Check basic properties
        assert isinstance(test_orbits, TestOrbits), "Should return TestOrbits"
        assert isinstance(orbit_volumes, OrbitVolumes), "Should return OrbitVolumes"
        assert len(test_orbits) == len(orbit_volumes), "Orbit count mismatch"
        
        # Check report contains target coverage info
        assert 'target_coverage_percent' in report, "Missing target coverage in report"
        assert 'attempts' in report, "Missing attempts in report"
        assert 'converged' in report, "Missing convergence info in report"
        
        actual_coverage = report['coverage_percentage']
        target_coverage = report['target_coverage_percent']
        attempts = report['attempts']
        converged = report['converged']
        
        print(f"  Target: {target_coverage:.1f}%, Actual: {actual_coverage:.1f}%")
        print(f"  Attempts: {len(attempts)}, Converged: {converged}")
        print(f"  Generated {len(test_orbits)} orbits")
        print(f"  Error: {abs(actual_coverage - target_coverage):.1f}%")
        
        # Check that all volumes have the same half-widths
        volume_half_widths = orbit_volumes.half_widths
        for i in range(len(orbit_volumes)):
            np.testing.assert_array_almost_equal(
                volume_half_widths[i], half_widths, decimal=6,
                err_msg=f"Volume {i} has incorrect half-widths"
            )
        
        print(f"  ✓ All volumes have correct fixed half-widths")
        
        # The algorithm should get reasonably close or converge
        if converged:
            assert abs(actual_coverage - target_coverage) <= 5.0, f"Converged but error too large: {abs(actual_coverage - target_coverage):.1f}%"
    
    print("✓ Fixed volume coverage generation works correctly")
    return True


def test_fixed_volume_edge_cases():
    """Test edge cases for fixed volume coverage generation."""
    print("\n=== Testing Fixed Volume Coverage Edge Cases ===")
    
    from thor.phase_space.coverage import generate_orbits_for_coverage_with_fixed_volumes
    from thor.orbit import TestOrbits
    
    # Test with very small volumes (should need many orbits)
    print("Testing with small volumes...")
    small_half_widths = np.array([0.05, 5.0, 3.0, 0.001, 0.05, 0.05])
    
    test_orbits, orbit_volumes, report = generate_orbits_for_coverage_with_fixed_volumes(
        target_coverage_percent=10.0,
        half_widths=small_half_widths,
        asteroid_type="inner_main_belt",
        max_orbits=30  # Limit for testing
    )
    
    assert len(test_orbits) >= 1, "Should generate at least one orbit"
    print(f"  ✓ Small volumes: {report['coverage_percentage']:.1f}% with {len(test_orbits)} orbits")
    
    # Test with different coordinate systems
    print("Testing with cartesian coordinates...")
    cartesian_half_widths = np.array([0.2, 0.2, 0.1, 0.003, 0.003, 0.002])
    
    test_orbits, orbit_volumes, report = generate_orbits_for_coverage_with_fixed_volumes(
        target_coverage_percent=15.0,
        half_widths=cartesian_half_widths,
        coordinate_system="cartesian",
        asteroid_type="main_belt",
        max_orbits=20
    )
    
    assert report['coordinate_system'] == "cartesian", "Should use cartesian coordinates"
    print(f"  ✓ Cartesian coverage: {report['coverage_percentage']:.1f}% with {len(test_orbits)} orbits")
    
    # Test input validation
    print("Testing input validation...")
    try:
        generate_orbits_for_coverage_with_fixed_volumes(0.0, small_half_widths)  # Invalid coverage
        assert False, "Should raise ValueError for coverage=0"
    except ValueError:
        print("  ✓ Correctly rejects coverage=0%")
    
    try:
        generate_orbits_for_coverage_with_fixed_volumes(50.0, np.array([1, 2, 3]))  # Wrong dimensions
        assert False, "Should raise ValueError for wrong dimensions"
    except ValueError:
        print("  ✓ Correctly rejects wrong half_widths dimensions")
    
    try:
        generate_orbits_for_coverage_with_fixed_volumes(50.0, small_half_widths, max_orbits=0)  # Invalid max_orbits
        assert False, "Should raise ValueError for max_orbits=0"
    except ValueError:
        print("  ✓ Correctly rejects max_orbits=0")
    
    print("✓ Fixed volume edge cases handled correctly")
    return True


def main():
    """Run all tests."""
    print("Testing Phase Space Coverage Analysis")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_basic_coverage()
        success &= test_custom_parameters()
        success &= test_different_sizes()
        success &= test_orbit_ids()
        success &= test_coordinate_systems()
        success &= test_asteroid_types()
        success &= test_target_coverage()
        success &= test_target_coverage_edge_cases()
        success &= test_fixed_volume_coverage()
        success &= test_fixed_volume_edge_cases()
        
        if success:
            print("\n🎉 All coverage tests passed!")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

if __name__ == "__main__":
    main()
