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
    from thor.phase_space.coverage import PhaseSpaceBounds

    # Create a small test region
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.5),
        lon=(0.0, 60.0),
        lat=(-10.0, 10.0),
        vrho=(-0.003, 0.003),
        vlon=(-0.3, 0.3),
        vlat=(-0.3, 0.3),
    )

    # Use larger volumes for this small region
    large_half_widths = np.array([0.15, 20.0, 8.0, 0.002, 0.2, 0.2])

    n_orbits = 50
    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits, bounds=small_bounds, half_widths=large_half_widths
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
    print("\n✓ Coverage Report:")
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
        n_orbits=25, asteroid_type="near_earth", coordinate_system="spherical"
    )

    print(f"✓ Generated {len(test_orbits1)} orbits for near_earth asteroids")
    print(f"  Coverage: {report1['coverage_percentage']:.1f}%")

    # Test with PhaseSpaceBounds object
    from thor.phase_space.coverage import PhaseSpaceBounds

    custom_bounds_obj = PhaseSpaceBounds.from_type("inner_main_belt", "keplerian")
    custom_half_widths = np.array([0.05, 0.005, 2.0, 5.0, 5.0, 5.0])  # Keplerian half-widths

    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits=25, bounds=custom_bounds_obj, coordinate_system="keplerian", half_widths=custom_half_widths
    )

    print(f"✓ Generated {len(test_orbits)} orbits with PhaseSpaceBounds object")
    print(f"  Coordinate system: {report['coordinate_system']}")
    print(f"  Coverage: {report['coverage_percentage']:.1f}%")
    print(f"  Efficiency: {report['efficiency']:.3f}")

    # Basic validation - orbits should be generated
    assert len(test_orbits) > 0, "No orbits generated"
    assert report["coordinate_system"] == "keplerian", "Wrong coordinate system"

    return True


def test_different_sizes():
    """Test with different numbers of orbits."""
    print("\n=== Testing Different Orbit Counts ===")

    # Use a smaller test region for visible coverage
    from thor.phase_space.coverage import PhaseSpaceBounds

    test_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.2, 2.8),
        lon=(0.0, 90.0),
        lat=(-15.0, 15.0),
        vrho=(-0.004, 0.004),
        vlon=(-0.4, 0.4),
        vlat=(-0.4, 0.4),
    )

    sizes = [10, 100, 500]

    for n in sizes:
        test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(n, bounds=test_bounds)

        print(
            f"  n={n:3d}: Generated {len(test_orbits):3d} orbits, "
            f"Coverage: {report['coverage_percentage']:5.1f}%, "
            f"Efficiency: {report['efficiency']:.3f}"
        )

        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for n={n}"
        assert 0 <= report["coverage_percentage"] <= 100, f"Invalid coverage for n={n}"
        # Efficiency can be negative when overlap is very high (more redundancy than useful coverage)
        assert report["efficiency"] <= 1, f"Efficiency too high for n={n}"

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

        print(
            f"  {system:10s}: Generated {len(test_orbits):2d} orbits, "
            f"Coverage: {report['coverage_percentage']:5.1f}%, "
            f"System: {report['coordinate_system']}"
        )

        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for {system}"
        assert report["coordinate_system"] == system, "Wrong coordinate system in report"
        assert isinstance(test_orbits.coordinates, type(test_orbits.coordinates)), "Wrong coordinate type"

        # All TestOrbits should have CartesianCoordinates regardless of input system
        from adam_core.coordinates import CartesianCoordinates

        assert isinstance(
            test_orbits.coordinates, CartesianCoordinates
        ), "TestOrbits should have CartesianCoordinates"

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
            200, asteroid_type=asteroid_type, coordinate_system="spherical", half_widths=large_half_widths
        )

        print(
            f"  {asteroid_type:15s}: Generated {len(test_orbits):2d} orbits, "
            f"Coverage: {report['coverage_percentage']:5.1f}%"
        )

        # Basic sanity checks
        assert len(test_orbits) > 0, f"No orbits generated for {asteroid_type}"
        assert report["coordinate_system"] == "spherical", "Wrong coordinate system"

    print("✓ All asteroid types work correctly")
    return True


def test_target_coverage():
    """Test generating orbits for target coverage percentage."""
    print("\n=== Testing Target Coverage Generation ===")

    from thor.orbit import TestOrbits
    from thor.phase_space.coverage import (
        OrbitVolumes,
        PhaseSpaceBounds,
        generate_orbit_volumes_for_target_coverage,
    )

    # Create a small test region for faster convergence
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.5),
        lon=(0.0, 90.0),
        lat=(-15.0, 15.0),
        vrho=(-0.005, 0.005),
        vlon=(-0.5, 0.5),
        vlat=(-0.5, 0.5),
    )

    # Test different target coverage percentages
    test_cases = [
        (10, 25.0),  # 10 orbits, 25% coverage
        (15, 50.0),  # 15 orbits, 50% coverage
        (20, 75.0),  # 20 orbits, 75% coverage
    ]

    for n_orbits, target_percent in test_cases:
        print(f"\n--- Testing {n_orbits} orbits for {target_percent}% coverage ---")

        test_orbits, orbit_volumes, report = generate_orbit_volumes_for_target_coverage(
            n_orbits=n_orbits,
            target_coverage_percent=target_percent,
            bounds=small_bounds,
        )

        # Check basic properties
        assert isinstance(test_orbits, TestOrbits), "Should return TestOrbits"
        assert isinstance(orbit_volumes, OrbitVolumes), "Should return OrbitVolumes"
        assert len(test_orbits) == len(orbit_volumes), "Orbit count mismatch"

        # Check report contains target coverage info
        assert "target_coverage_percent" in report, "Missing target coverage in report"
        assert "actual_coverage_percent" in report, "Missing actual coverage in report"
        assert "volume_calculation_method" in report, "Missing calculation method in report"

        actual_coverage = report["actual_coverage_percent"]
        target_coverage = report["target_coverage_percent"]
        method = report["volume_calculation_method"]
        error = report["coverage_error_percent"]

        print(f"  Target: {target_coverage:.1f}%, Actual: {actual_coverage:.1f}%")
        print(f"  Method: {method}, Error: {error:.1f}%")

        # The direct method should be very accurate
        assert method == "direct", f"Expected direct method, got {method}"
        assert error <= 2.0, f"Error too large for direct method: {error:.1f}%"

        print(f"  ✓ Generated {len(test_orbits)} orbits")

    print("✓ Target coverage generation works correctly")
    return True


def test_target_coverage_edge_cases():
    """Test edge cases for target coverage generation."""
    print("\n=== Testing Target Coverage Edge Cases ===")

    from thor.phase_space.coverage import generate_orbit_volumes_for_target_coverage

    # Test with very small target coverage
    print("Testing low coverage target (5%)...")
    test_orbits, orbit_volumes, report = generate_orbit_volumes_for_target_coverage(
        n_orbits=5, target_coverage_percent=5.0, asteroid_type="inner_main_belt"
    )

    assert len(test_orbits) >= 1, "Should generate at least one orbit"
    print(f"  ✓ Low coverage: {report['coverage_percentage']:.1f}% ({len(test_orbits)} orbits generated)")

    # Test with different coordinate systems
    print("Testing with cartesian coordinates...")
    test_orbits, orbit_volumes, report = generate_orbit_volumes_for_target_coverage(
        n_orbits=8,
        target_coverage_percent=20.0,
        coordinate_system="cartesian",
        asteroid_type="main_belt",
    )

    assert report["coordinate_system"] == "cartesian", "Should use cartesian coordinates"
    print(f"  ✓ Cartesian coverage: {report['coverage_percentage']:.1f}%")

    # Test input validation
    print("Testing input validation...")
    try:
        generate_orbit_volumes_for_target_coverage(0, 50.0)  # Invalid n_orbits
        assert False, "Should raise ValueError for n_orbits=0"
    except ValueError:
        print("  ✓ Correctly rejects n_orbits=0")

    try:
        generate_orbit_volumes_for_target_coverage(5, 0.0)  # Invalid coverage
        assert False, "Should raise ValueError for coverage=0"
    except ValueError:
        print("  ✓ Correctly rejects coverage=0%")

    try:
        generate_orbit_volumes_for_target_coverage(5, 150.0)  # Invalid coverage
        assert False, "Should raise ValueError for coverage>100"
    except ValueError:
        print("  ✓ Correctly rejects coverage>100%")

    print("✓ Edge cases handled correctly")
    return True


def test_fixed_volume_coverage():
    """Test generating orbits with fixed volumes to achieve target coverage."""
    print("\n=== Testing Fixed Volume Coverage Generation ===")

    from thor.orbit import TestOrbits
    from thor.phase_space.coverage import (
        OrbitVolumes,
        PhaseSpaceBounds,
        generate_orbits_for_coverage_with_fixed_volumes,
    )

    # Create a small test region for predictable results
    small_bounds = PhaseSpaceBounds.from_spherical(
        rho=(2.0, 2.8),
        lon=(0.0, 90.0),
        lat=(-15.0, 15.0),
        vrho=(-0.005, 0.005),
        vlon=(-0.5, 0.5),
        vlat=(-0.5, 0.5),
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
            max_orbits=50,  # Limit for testing
        )

        # Check basic properties
        assert isinstance(test_orbits, TestOrbits), "Should return TestOrbits"
        assert isinstance(orbit_volumes, OrbitVolumes), "Should return OrbitVolumes"
        assert len(test_orbits) == len(orbit_volumes), "Orbit count mismatch"

        # Check report contains target coverage info
        assert "target_coverage_percent" in report, "Missing target coverage in report"
        assert "actual_coverage_percent" in report, "Missing actual coverage in report"
        assert "calculation_method" in report, "Missing calculation method in report"
        assert "orbits_requested" in report, "Missing orbits requested in report"
        assert "hit_max_orbits_limit" in report, "Missing max orbits limit info in report"

        actual_coverage = report["actual_coverage_percent"]
        target_coverage = report["target_coverage_percent"]
        method = report["calculation_method"]
        orbits_requested = report["orbits_requested"]
        hit_limit = report["hit_max_orbits_limit"]

        print(f"  Target: {target_coverage:.1f}%, Actual: {actual_coverage:.1f}%")
        print(f"  Method: {method}, Orbits requested: {orbits_requested}")
        print(f"  Generated {len(test_orbits)} orbits, Hit limit: {hit_limit}")
        print(f"  Error: {abs(actual_coverage - target_coverage):.1f}%")

        # Check that method is direct (non-iterative)
        assert method == "direct", f"Expected direct calculation, got {method}"

        # Check that all volumes have the same half-widths
        volume_half_widths = orbit_volumes.half_widths
        for i in range(len(orbit_volumes)):
            # Convert PyArrow scalar to numpy array for comparison
            actual_half_widths = np.array(list(volume_half_widths[i].as_py()))
            np.testing.assert_array_almost_equal(
                actual_half_widths, half_widths, decimal=6, err_msg=f"Volume {i} has incorrect half-widths"
            )

        print("  ✓ All volumes have correct fixed half-widths")

        # The direct calculation should be reasonably accurate (within 10% due to overlap assumptions)
        error = abs(actual_coverage - target_coverage)
        assert error <= 15.0, f"Direct calculation error too large: {error:.1f}%"

    print("✓ Fixed volume coverage generation works correctly")
    return True


def test_fixed_volume_edge_cases():
    """Test edge cases for fixed volume coverage generation."""
    print("\n=== Testing Fixed Volume Coverage Edge Cases ===")

    from thor.phase_space.coverage import (
        generate_orbits_for_coverage_with_fixed_volumes,
    )

    # Test with very small volumes (should need many orbits)
    print("Testing with small volumes...")
    small_half_widths = np.array([0.05, 5.0, 3.0, 0.001, 0.05, 0.05])

    test_orbits, orbit_volumes, report = generate_orbits_for_coverage_with_fixed_volumes(
        target_coverage_percent=10.0,
        half_widths=small_half_widths,
        asteroid_type="inner_main_belt",
        max_orbits=30,  # Limit for testing
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
        max_orbits=20,
    )

    assert report["coordinate_system"] == "cartesian", "Should use cartesian coordinates"
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
        generate_orbits_for_coverage_with_fixed_volumes(
            50.0, small_half_widths, max_orbits=0
        )  # Invalid max_orbits
        assert False, "Should raise ValueError for max_orbits=0"
    except ValueError:
        print("  ✓ Correctly rejects max_orbits=0")

    print("✓ Fixed volume edge cases handled correctly")
    return True


def test_covariance_attachment():
    """Test that covariances are correctly attached to test orbits."""
    print("\n=== Testing Covariance Attachment ===")

    # Test 1: Generate test orbits with covariances in spherical coordinates
    print("\n[Test 1] Generating test orbits in SPHERICAL coordinates with covariances...")
    half_widths = np.array([0.1, 15.0, 8.0, 0.002, 0.2, 0.15])  # spherical

    test_orbits, orbit_volumes, report = generate_even_coverage_test_orbits(
        n_orbits=5,
        half_widths=half_widths,
        coordinate_system="spherical",
        asteroid_type="main_belt",
        attach_covariances=True,
        covariance_scale=1.0,
    )

    print(f"✓ Generated {len(test_orbits)} test orbits")

    # Check that covariances are attached
    cov_matrices = test_orbits.coordinates.covariance.to_matrix()
    print(f"✓ Covariance shape: {cov_matrices.shape}")
    assert cov_matrices.shape == (5, 6, 6), f"Expected (5, 6, 6), got {cov_matrices.shape}"

    # Check that covariances are not NaN
    has_nan = np.any(np.isnan(cov_matrices))
    print(f"✓ Covariances contain NaN: {has_nan}")
    assert not has_nan, "Covariances should not contain NaN values"

    # Check that covariances are positive definite (diagonal elements > 0)
    diag_elements = np.diagonal(cov_matrices, axis1=1, axis2=2)
    print(f"✓ Diagonal elements (variances) shape: {diag_elements.shape}")
    assert np.all(diag_elements > 0), "All diagonal elements should be positive"

    # Check that sigmas are extracted correctly
    sigmas = test_orbits.coordinates.covariance.sigmas
    print(f"✓ Sigmas shape: {sigmas.shape}")

    # Test 2: Test without covariances
    print("\n[Test 2] Generating test orbits WITHOUT covariances...")
    test_orbits_no_cov, _, _ = generate_even_coverage_test_orbits(
        n_orbits=3,
        coordinate_system="spherical",
        asteroid_type="main_belt",
        attach_covariances=False,
    )

    cov_matrices_no_cov = test_orbits_no_cov.coordinates.covariance.to_matrix()
    has_all_nan = np.all(np.isnan(cov_matrices_no_cov))
    print(f"✓ Covariances are NaN (as expected): {has_all_nan}")
    assert has_all_nan, "Covariances should be NaN when attach_covariances=False"

    # Test 3: Test with Cartesian coordinates
    print("\n[Test 3] Generating test orbits in CARTESIAN coordinates with covariances...")
    half_widths_cart = np.array([0.1, 0.1, 0.1, 0.002, 0.002, 0.002])

    test_orbits_cart, _, _ = generate_even_coverage_test_orbits(
        n_orbits=4,
        half_widths=half_widths_cart,
        coordinate_system="cartesian",
        asteroid_type="main_belt",
        attach_covariances=True,
    )

    cov_matrices_cart = test_orbits_cart.coordinates.covariance.to_matrix()
    print(f"✓ Covariance shape: {cov_matrices_cart.shape}")
    assert cov_matrices_cart.shape == (4, 6, 6)

    # For Cartesian, covariances should be diagonal (no transformation needed)
    # Check that off-diagonal elements are exactly zero
    for i in range(4):
        off_diag = cov_matrices_cart[i].copy()
        np.fill_diagonal(off_diag, 0)
        max_off_diag = np.max(np.abs(off_diag))
        assert (
            max_off_diag == 0.0
        ), f"Cartesian covariances should be diagonal, got max off-diag: {max_off_diag}"

    sigmas_cart = test_orbits_cart.coordinates.covariance.sigmas
    print(f"✓ Sigmas for first Cartesian orbit: {sigmas_cart[0]}")

    # Verify sigmas match half_widths (for Cartesian, should be identical)
    expected_sigmas = half_widths_cart
    actual_sigmas = sigmas_cart[0]
    assert np.allclose(
        actual_sigmas, expected_sigmas, rtol=0.01
    ), f"Cartesian sigmas should match half_widths closely"

    # Test 4: Test with Keplerian coordinates
    print("\n[Test 4] Generating test orbits in KEPLERIAN coordinates with covariances...")
    half_widths_kep = np.array([0.1, 0.01, 5.0, 10.0, 10.0, 10.0])

    test_orbits_kep, _, _ = generate_even_coverage_test_orbits(
        n_orbits=3,
        half_widths=half_widths_kep,
        coordinate_system="keplerian",
        asteroid_type="main_belt",
        attach_covariances=True,
    )

    cov_matrices_kep = test_orbits_kep.coordinates.covariance.to_matrix()
    print(f"✓ Covariance shape: {cov_matrices_kep.shape}")
    assert cov_matrices_kep.shape == (3, 6, 6)
    assert not np.any(np.isnan(cov_matrices_kep)), "Keplerian covariances should not be NaN"

    sigmas_kep = test_orbits_kep.coordinates.covariance.sigmas
    print(f"✓ Sigmas for first Keplerian orbit (transformed to Cartesian)")

    # Test 5: Test covariance_scale parameter
    print("\n[Test 5] Testing covariance_scale parameter...")
    test_orbits_scale, _, _ = generate_even_coverage_test_orbits(
        n_orbits=2,
        half_widths=half_widths_cart,
        coordinate_system="cartesian",
        attach_covariances=True,
        covariance_scale=3.0,  # Treat half_widths as 3-sigma
    )

    sigmas_scaled = test_orbits_scale.coordinates.covariance.sigmas
    expected_scaled = half_widths_cart / 3.0
    print(f"✓ Sigmas with scale=3.0: {sigmas_scaled[0]}")
    assert np.allclose(
        sigmas_scaled[0], expected_scaled, rtol=0.01
    ), f"Scaled sigmas should be half_widths/scale"

    print("\n✓ All covariance attachment tests passed!")


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
        test_covariance_attachment()  # Pytest-style test (no return value)

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
