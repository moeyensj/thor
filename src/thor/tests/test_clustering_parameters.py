import numpy as np
import pytest
from adam_core.coordinates import CartesianCoordinates, Origin, SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbits import Ephemeris
from adam_core.time import Timestamp

from ..clusters import calculate_clustering_parameters_from_covariance
from ..orbit import TestOrbitEphemeris
from ..projections.covariances import ProjectionCovariances
from ..projections.gnomonic import GnomonicCoordinates
from ..range_and_transform import TransformedDetections


def _make_test_data(
    n_times=4,
    sigma_theta_x=1e-3,
    sigma_theta_y=2e-3,
    sigma_vtheta_x=1e-4,
    sigma_vtheta_y=1e-4,
    n_obs=None,
    cov_matrices=None,
):
    """
    Build minimal TestOrbitEphemeris and TransformedDetections for testing
    calculate_clustering_parameters_from_covariance.

    Parameters
    ----------
    n_times : int
        Number of ephemeris/observation time steps.
    sigma_theta_x, sigma_theta_y : float
        Positional standard deviations (degrees) for diagonal covariance.
    sigma_vtheta_x, sigma_vtheta_y : float
        Velocity standard deviations (deg/day) for diagonal covariance.
    n_obs : int, optional
        Number of transformed detections. If None, defaults to n_times * 100.
    cov_matrices : np.ndarray, optional
        Custom (n_times, 4, 4) covariance matrices. If provided, overrides
        the sigma parameters.
    """
    if n_obs is None:
        n_obs = n_times * 100

    mjd0 = 59000.0
    times_mjd = mjd0 + np.arange(n_times, dtype=np.float64)
    t = Timestamp.from_mjd(times_mjd, scale="utc")

    # Build covariance matrices
    if cov_matrices is None:
        cov = np.zeros((n_times, 4, 4), dtype=np.float64)
        cov[:, 0, 0] = sigma_theta_x**2
        cov[:, 1, 1] = sigma_theta_y**2
        cov[:, 2, 2] = sigma_vtheta_x**2
        cov[:, 3, 3] = sigma_vtheta_y**2
    else:
        cov = cov_matrices

    covariances = ProjectionCovariances.from_matrix(cov)
    origin = Origin.from_kwargs(code=["SUN"] * n_times)

    gnomonic = GnomonicCoordinates.from_kwargs(
        time=t,
        theta_x=np.zeros(n_times),
        theta_y=np.zeros(n_times),
        covariance=covariances,
        origin=origin,
        frame="testorbit",
    )

    # Build minimal Ephemeris (required by TestOrbitEphemeris but not used
    # by calculate_clustering_parameters_from_covariance)
    spherical = SphericalCoordinates.from_kwargs(
        lon=np.zeros(n_times),
        lat=np.zeros(n_times),
        time=t,
        origin=origin,
        frame="equatorial",
    )
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=["test_orbit"] * n_times,
        coordinates=spherical,
    )

    # Build minimal Observers
    observer_coords = CartesianCoordinates.from_kwargs(
        x=np.ones(n_times),
        y=np.zeros(n_times),
        z=np.zeros(n_times),
        vx=np.zeros(n_times),
        vy=np.zeros(n_times),
        vz=np.zeros(n_times),
        time=t,
        origin=origin,
        frame="ecliptic",
    )
    observers = Observers.from_kwargs(
        code=["500"] * n_times,
        coordinates=observer_coords,
    )

    # Build TestOrbitEphemeris
    rotation_matrices = [np.eye(6).flatten().tolist() for _ in range(n_times)]
    test_orbit_ephemeris = TestOrbitEphemeris.from_kwargs(
        id=[f"state_{i}" for i in range(n_times)],
        test_orbit_id=["test_orbit"] * n_times,
        ephemeris=ephemeris,
        observer=observers,
        gnomonic=gnomonic,
        gnomonic_rotation_matrix=rotation_matrices,
    )

    # Build TransformedDetections (spread across the same time range)
    obs_times = np.linspace(times_mjd[0], times_mjd[-1], n_obs)
    obs_t = Timestamp.from_mjd(obs_times, scale="utc")
    obs_cov = np.zeros((n_obs, 4, 4), dtype=np.float64)
    obs_cov[:, 0, 0] = sigma_theta_x**2
    obs_cov[:, 1, 1] = sigma_theta_y**2
    obs_cov[:, 2, 2] = sigma_vtheta_x**2
    obs_cov[:, 3, 3] = sigma_vtheta_y**2
    obs_covariances = ProjectionCovariances.from_matrix(obs_cov)
    obs_origin = Origin.from_kwargs(code=["SUN"] * n_obs)

    obs_coords = GnomonicCoordinates.from_kwargs(
        time=obs_t,
        theta_x=np.random.default_rng(42).normal(scale=sigma_theta_x, size=n_obs),
        theta_y=np.random.default_rng(43).normal(scale=sigma_theta_y, size=n_obs),
        covariance=obs_covariances,
        origin=obs_origin,
        frame="testorbit",
    )
    transformed_detections = TransformedDetections.from_kwargs(
        id=[f"obs_{i}" for i in range(n_obs)],
        test_orbit_id=["test_orbit"] * n_obs,
        night=np.repeat(np.arange(n_times), n_obs // n_times + 1)[:n_obs].astype(np.int64),
        coordinates=obs_coords,
        state_id=[f"state_{i % n_times}" for i in range(n_obs)],
    )

    return test_orbit_ephemeris, transformed_detections


class TestRadiusFromEigenvalues:
    """Test that the radius lower bound is driven by positional covariance eigenvalues."""

    def test_basic_diagonal_covariance(self):
        """With diagonal covariances, max eigenvalue = max(sigma_x^2, sigma_y^2).

        Pass astrometric_precision=0 to isolate the orbital contribution so that
        radius_combined = multiplier * sqrt(0^2 + sigma_orbital^2) = multiplier * sigma_y.
        """
        sigma_x = 1e-3  # deg
        sigma_y = 2e-3  # deg
        multiplier = 5.0

        ephem, detections = _make_test_data(
            sigma_theta_x=sigma_x,
            sigma_theta_y=sigma_y,
        )
        _, _, radius, metadata = calculate_clustering_parameters_from_covariance(
            ephem,
            detections,
            radius_multiplier=multiplier,
            max_radius=1.0,  # large ceiling so it doesn't clip
            astrometric_precision=0.0,  # isolate orbital contribution
        )

        expected_radius_combined = multiplier * sigma_y  # sigma_y > sigma_x, sigma_astro=0
        assert metadata["max_sigma_pos"] == pytest.approx(sigma_y, rel=1e-6)
        assert metadata["radius_combined"] == pytest.approx(expected_radius_combined, rel=1e-6)
        # Radius should be at least radius_combined (could be capped by r_upper)
        assert radius >= expected_radius_combined or radius == pytest.approx(metadata["r_upper"], rel=1e-6)

    def test_multiplier_scaling(self):
        """Doubling the multiplier should double radius_combined."""
        ephem, detections = _make_test_data(sigma_theta_x=1e-3, sigma_theta_y=1e-3)

        _, _, _, meta_5 = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=5.0, max_radius=1.0, astrometric_precision=0.0
        )
        _, _, _, meta_10 = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=10.0, max_radius=1.0, astrometric_precision=0.0
        )

        assert meta_10["radius_combined"] == pytest.approx(2.0 * meta_5["radius_combined"], rel=1e-6)

    def test_off_diagonal_covariance(self):
        """With correlated covariance, eigenvalues differ from diagonal elements."""
        n_times = 4
        cov = np.zeros((n_times, 4, 4), dtype=np.float64)
        # Positional covariance with correlation:
        # [[1e-6, 0.8e-6], [0.8e-6, 1e-6]]
        # Eigenvalues: 1.8e-6, 0.2e-6
        cov[:, 0, 0] = 1e-6
        cov[:, 1, 1] = 1e-6
        cov[:, 0, 1] = 0.8e-6
        cov[:, 1, 0] = 0.8e-6
        cov[:, 2, 2] = 1e-8
        cov[:, 3, 3] = 1e-8

        ephem, detections = _make_test_data(cov_matrices=cov)
        _, _, _, metadata = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=5.0, max_radius=1.0
        )

        expected_max_sigma = np.sqrt(1.8e-6)
        assert metadata["max_sigma_pos"] == pytest.approx(expected_max_sigma, rel=1e-4)


class TestRadiusClipping:
    """Test that radius is clipped to [min_radius, max_radius]."""

    def test_clipped_to_min_radius(self):
        """Very small covariance → radius should equal min_radius."""
        min_rad = 1 / 3600  # 1 arcsec
        ephem, detections = _make_test_data(
            sigma_theta_x=1e-8,  # extremely small
            sigma_theta_y=1e-8,
        )
        _, _, radius, _ = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=5.0, min_radius=min_rad, max_radius=1.0
        )

        assert radius == pytest.approx(min_rad, rel=1e-6)

    def test_clipped_to_max_radius(self):
        """Very large covariance → radius should equal max_radius."""
        max_rad = 0.01
        ephem, detections = _make_test_data(
            sigma_theta_x=1.0,  # 1 degree — huge
            sigma_theta_y=1.0,
            n_obs=10,  # few obs so density doesn't cap first
        )
        _, _, radius, _ = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=5.0, max_radius=max_rad
        )

        assert radius == pytest.approx(max_rad, rel=1e-6)


class TestDensityUpperBound:
    """Test that detection density within the observation footprint caps the radius."""

    def test_density_caps_radius(self):
        """With many observations in a small footprint, density upper bound
        should cap the radius below what r_lower would give."""
        # Small positional scatter + many observations = high density = small r_upper
        ephem, detections = _make_test_data(
            sigma_theta_x=1e-3,
            sigma_theta_y=1e-3,
            sigma_vtheta_x=1e-4,
            sigma_vtheta_y=1e-4,
            n_obs=100000,  # very dense
        )
        _, _, radius, metadata = calculate_clustering_parameters_from_covariance(
            ephem,
            detections,
            radius_multiplier=50.0,  # would give huge radius_combined
            density_multiplier=2.5,
            min_radius=1e-10,  # tiny floor so it doesn't interfere
            max_radius=1.0,
            astrometric_precision=0.0,  # isolate orbital contribution
        )

        # radius_combined = 50 * 1e-3 = 0.05 deg (with astrometric_precision=0)
        # r_upper should be much less due to high density in the observation footprint
        assert metadata["r_upper"] < metadata["radius_combined"]
        assert radius == pytest.approx(metadata["r_upper"], rel=1e-6)
        # obs_area should be the bounding box of the observations
        assert metadata["obs_area"] > 0

    def test_density_uses_observation_footprint(self):
        """Density should be n_obs / obs_area where obs_area is the bounding box
        of the actual observation positions."""
        ephem, detections = _make_test_data(
            sigma_theta_x=1e-3,
            sigma_theta_y=1e-3,
            n_obs=10000,
        )
        _, _, _, metadata = calculate_clustering_parameters_from_covariance(ephem, detections, max_radius=1.0)

        # Verify density = n_obs / obs_area
        expected_density = metadata["n_obs"] / metadata["obs_area"]
        assert metadata["density"] == pytest.approx(expected_density, rel=1e-6)

    def test_density_multiplier_effect(self):
        """Higher density_multiplier allows larger radii in dense fields."""
        ephem, detections = _make_test_data(n_obs=10000)

        _, _, _, meta_low = calculate_clustering_parameters_from_covariance(
            ephem, detections, density_multiplier=1.0, max_radius=1.0
        )
        _, _, _, meta_high = calculate_clustering_parameters_from_covariance(
            ephem, detections, density_multiplier=5.0, max_radius=1.0
        )

        assert meta_high["r_upper"] == pytest.approx(5.0 * meta_low["r_upper"], rel=1e-6)


class TestNaNHandling:
    """Test handling of NaN and invalid covariance matrices."""

    def test_some_nan_covariances_filtered(self):
        """Function should use only valid covariance matrices, skipping NaNs."""
        n_times = 4
        cov = np.zeros((n_times, 4, 4), dtype=np.float64)
        cov[:, 0, 0] = 1e-6
        cov[:, 1, 1] = 4e-6
        cov[:, 2, 2] = 1e-8
        cov[:, 3, 3] = 1e-8
        # Make the last two covariance matrices NaN
        cov[2:, :, :] = np.nan

        ephem, detections = _make_test_data(cov_matrices=cov)
        _, _, radius, metadata = calculate_clustering_parameters_from_covariance(
            ephem, detections, radius_multiplier=5.0, max_radius=1.0
        )

        # Should use max eigenvalue from the valid covariances only
        expected_max_sigma = np.sqrt(4e-6)  # sigma_y = 2e-3
        assert metadata["max_sigma_pos"] == pytest.approx(expected_max_sigma, rel=1e-4)
        assert radius > 0

    def test_all_nan_raises(self):
        """All NaN covariances should raise ValueError."""
        n_times = 4
        cov = np.full((n_times, 4, 4), np.nan, dtype=np.float64)

        ephem, detections = _make_test_data(cov_matrices=cov)
        with pytest.raises(ValueError, match="All covariance values are NaN"):
            calculate_clustering_parameters_from_covariance(ephem, detections, radius_multiplier=5.0)


class TestVelocityGrid:
    """Test that the velocity grid calculation is correct."""

    def test_grid_shape_and_filtering(self):
        """Velocity grid should be filtered to Mahalanobis ellipse."""
        ephem, detections = _make_test_data(
            sigma_vtheta_x=0.01,
            sigma_vtheta_y=0.005,
        )
        vx, vy, _, metadata = calculate_clustering_parameters_from_covariance(
            ephem, detections, mahalanobis_distance=3.0, max_radius=1.0
        )

        # vx and vy should have same length
        assert len(vx) == len(vy)
        assert len(vx) > 0
        assert len(vx) == metadata["n_velocity_points"]

        # All points should be within the Mahalanobis distance ellipse
        sigma_vx = metadata["sigma_vx"]
        sigma_vy = metadata["sigma_vy"]
        # Simple check: all vx within 3*sigma_vx, all vy within 3*sigma_vy
        assert np.all(np.abs(vx) <= 3.0 * sigma_vx + 1e-10)
        assert np.all(np.abs(vy) <= 3.0 * sigma_vy + 1e-10)

    def test_zero_velocity_included(self):
        """The zero velocity point should always be in the grid."""
        ephem, detections = _make_test_data()
        vx, vy, _, _ = calculate_clustering_parameters_from_covariance(ephem, detections, max_radius=1.0)

        # At least one point should be at or very near (0, 0)
        distances = np.sqrt(vx**2 + vy**2)
        assert np.min(distances) < 1e-6


class TestMetadataKeys:
    """Test that metadata contains the expected keys."""

    def test_new_keys_present(self):
        ephem, detections = _make_test_data()
        _, _, _, metadata = calculate_clustering_parameters_from_covariance(ephem, detections, max_radius=1.0)

        expected_keys = {
            "n_obs",
            "n_times",
            "dt_arc",
            "sigma_astro_deg",
            "sigma_orbital_deg",
            "max_sigma_pos",
            "radius_multiplier",
            "radius_combined",
            "obs_area",
            "obs_extent_x",
            "obs_extent_y",
            "density",
            "density_multiplier",
            "r_sep",
            "r_upper",
            "radius",
            "min_radius",
            "max_radius",
            "mahalanobis_distance",
            "sigma_vx",
            "sigma_vy",
            "n_vx_bins",
            "n_vy_bins",
            "n_velocity_points",
            "dv_x",
            "dv_y",
        }

        for key in expected_keys:
            assert key in metadata, f"Missing expected metadata key: {key}"

    def test_removed_keys_absent(self):
        ephem, detections = _make_test_data()
        _, _, _, metadata = calculate_clustering_parameters_from_covariance(ephem, detections, max_radius=1.0)

        removed_keys = {
            "radius_candidate",
            "epsilon_min",
            "epsilon_max",
            "epsilon_astro",
            "epsilon_bin",
        }

        for key in removed_keys:
            assert key not in metadata, f"Removed key still present: {key}"
