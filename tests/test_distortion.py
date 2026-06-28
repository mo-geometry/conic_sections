"""Tests for radial lens distortion models.

Covers all five fish-eye projection models plus the pinhole baseline:
- LUT generation and shape
- Monotonicity of r(θ)
- Known analytical values
- Round-trip consistency (distort → undistort recovers original rays)
- Edge cases: optical axis, extreme field angles, zero radius
- Polynomial monotonicity validation (bad coefficients raise ValueError)
- Full pipeline: spherical_rays_to_pixel_coords / pixel_coords_to_spherical_rays
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from conic_sections.core.distortion import (
    DistortionCoeffs,
    LensModel,
    build_distortion_lut,
    distort_points,
    pixel_coords_to_spherical_rays,
    spherical_rays_to_pixel_coords,
    undistort_points,
)

# All non-pinhole fish-eye models (pinhole has a limited domain, tested separately)
FISHEYE_MODELS = [
    LensModel.EQUIDISTANT,
    LensModel.EQUISOLID,
    LensModel.STEREOGRAPHIC,
    LensModel.ORTHOGRAPHIC,
    LensModel.POLYNOMIAL,
]

ALL_MODELS = [LensModel.PINHOLE, *FISHEYE_MODELS]


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------


class TestBuildDistortionLUT:
    """Tests for build_distortion_lut."""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_lut_shape(self, model: LensModel) -> None:
        lut = build_distortion_lut(model)
        assert lut.theta.shape == (4096,)
        assert lut.r.shape == (4096,)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_lut_starts_at_zero(self, model: LensModel) -> None:
        lut = build_distortion_lut(model)
        assert lut.theta[0] == pytest.approx(0.0)
        assert lut.r[0] == pytest.approx(0.0)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_lut_model_stored(self, model: LensModel) -> None:
        lut = build_distortion_lut(model)
        assert lut.model == model

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_lut_monotonically_increasing(self, model: LensModel) -> None:
        """r(θ) must be monotonically increasing for a valid lens model."""
        lut = build_distortion_lut(model)
        dr = np.diff(lut.r)
        # Allow tiny floating point dips but not real inversions
        assert dr.min() >= -1e-12, f"r(θ) not monotonic for {model.value}"

    def test_custom_n_points(self) -> None:
        lut = build_distortion_lut(LensModel.EQUIDISTANT, n_points=256)
        assert lut.theta.shape == (256,)

    def test_polynomial_with_coefficients(self) -> None:
        coeffs = DistortionCoeffs(k2=0.01, k3=-0.005, k4=0.001)
        lut = build_distortion_lut(LensModel.POLYNOMIAL, coeffs=coeffs)
        # At θ = 0, r should be 0
        assert lut.r[0] == pytest.approx(0.0)
        # Should complete without error (monotonic)
        assert len(lut.r) == 4096

    def test_polynomial_bad_coefficients_raises(self) -> None:
        """Non-monotonic polynomial coefficients should raise ValueError."""
        # Large negative k2 will cause r to decrease at some θ
        bad_coeffs = DistortionCoeffs(k2=-5.0, k3=0.0, k4=0.0)
        with pytest.raises(ValueError, match="monotonically increasing"):
            build_distortion_lut(LensModel.POLYNOMIAL, coeffs=bad_coeffs)


# ---------------------------------------------------------------------------
# Known analytical values
# ---------------------------------------------------------------------------


class TestKnownValues:
    """Verify r(θ) matches the analytical formula at specific field angles."""

    def test_equidistant_identity(self) -> None:
        """Equidistant: r = θ at all angles."""
        lut = build_distortion_lut(LensModel.EQUIDISTANT)
        # Check at θ = 1.0 radian
        r = float(np.interp(1.0, lut.theta, lut.r))
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_equisolid_at_pi_over_3(self) -> None:
        """Equisolid: r(π/3) = 2·sin(π/6) = 1.0."""
        lut = build_distortion_lut(LensModel.EQUISOLID)
        r = float(np.interp(np.pi / 3, lut.theta, lut.r))
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_stereographic_at_pi_over_4(self) -> None:
        """Stereographic: r(π/4) = 2·tan(π/8)."""
        lut = build_distortion_lut(LensModel.STEREOGRAPHIC)
        expected = 2.0 * np.tan(np.pi / 8)
        r = float(np.interp(np.pi / 4, lut.theta, lut.r))
        assert r == pytest.approx(expected, abs=1e-4)

    def test_orthographic_at_pi_over_6(self) -> None:
        """Orthographic: r(π/6) ≈ sin(π/6) = 0.5."""
        lut = build_distortion_lut(LensModel.ORTHOGRAPHIC)
        r = float(np.interp(np.pi / 6, lut.theta, lut.r))
        assert r == pytest.approx(0.5, abs=0.01)

    def test_pinhole_at_pi_over_4(self) -> None:
        """Pinhole: r(π/4) = tan(π/4) = 1.0."""
        lut = build_distortion_lut(LensModel.PINHOLE)
        r = float(np.interp(np.pi / 4, lut.theta, lut.r))
        assert r == pytest.approx(1.0, abs=1e-3)

    def test_polynomial_defaults_match_equidistant(self) -> None:
        """Polynomial with zero coefficients is the identity: r = θ."""
        lut = build_distortion_lut(LensModel.POLYNOMIAL)
        # At θ = 1.5 radians
        r = float(np.interp(1.5, lut.theta, lut.r))
        assert r == pytest.approx(1.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Forward projection (distort_points)
# ---------------------------------------------------------------------------


class TestDistortPoints:
    """Tests for distort_points: unit rays → image plane."""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_on_axis_ray_projects_to_origin(self, model: LensModel) -> None:
        """A ray along +Z (θ=0) should map to (0, 0, 1) on the image plane."""
        lut = build_distortion_lut(model)
        xyz = np.array([[0.0, 0.0, 1.0]])
        uv1 = distort_points(xyz, lut)
        assert uv1[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert uv1[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert uv1[0, 2] == pytest.approx(1.0)

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_output_shape(self, model: LensModel) -> None:
        lut = build_distortion_lut(model)
        n = 50
        xyz = np.random.default_rng(42).normal(size=(n, 3))
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
        uv1 = distort_points(xyz, lut)
        assert uv1.shape == (n, 3)

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_third_column_is_one(self, model: LensModel) -> None:
        lut = build_distortion_lut(model)
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        uv1 = distort_points(xyz, lut)
        np.testing.assert_array_almost_equal(uv1[:, 2], 1.0)

    def test_equidistant_45_degree_ray(self) -> None:
        """A ray at 45° in the XZ plane: θ=π/4, φ=0, so r=π/4, u=π/4, v=0."""
        lut = build_distortion_lut(LensModel.EQUIDISTANT)
        angle = np.pi / 4
        xyz = np.array([[np.sin(angle), 0.0, np.cos(angle)]])
        uv1 = distort_points(xyz, lut)
        assert uv1[0, 0] == pytest.approx(np.pi / 4, abs=1e-3)
        assert uv1[0, 1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Inverse projection (undistort_points)
# ---------------------------------------------------------------------------


class TestUndistortPoints:
    """Tests for undistort_points: image plane → unit rays."""

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_origin_maps_to_z_axis(self, model: LensModel) -> None:
        """Image-plane origin (0,0,1) should map to the +Z ray."""
        lut = build_distortion_lut(model)
        uv1 = np.array([[0.0, 0.0, 1.0]])
        xyz = undistort_points(uv1, lut)
        assert xyz[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert xyz[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert xyz[0, 2] == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_output_is_unit_vectors(self, model: LensModel) -> None:
        """Undistorted rays should lie on the unit sphere."""
        lut = build_distortion_lut(model)
        uv1 = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [0.0, 0.3, 1.0],
                [0.2, 0.2, 1.0],
            ]
        )
        xyz = undistort_points(uv1, lut)
        norms = np.linalg.norm(xyz, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=10)


# ---------------------------------------------------------------------------
# Round-trip consistency: distort → undistort → original
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that distort followed by undistort recovers the original rays."""

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_round_trip_random_rays(self, model: LensModel) -> None:
        """Random unit rays: distort → undistort should recover the original."""
        lut = build_distortion_lut(model)
        rng = np.random.default_rng(123)
        xyz = rng.normal(size=(100, 3))
        # Keep rays in the forward hemisphere (z > 0) for all models
        xyz[:, 2] = np.abs(xyz[:, 2]) + 0.1
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        uv1 = distort_points(xyz, lut)
        xyz_recovered = undistort_points(uv1, lut)

        np.testing.assert_array_almost_equal(xyz, xyz_recovered, decimal=4)

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_round_trip_on_axis(self, model: LensModel) -> None:
        """The optical axis ray should survive the round trip exactly."""
        lut = build_distortion_lut(model)
        xyz = np.array([[0.0, 0.0, 1.0]])

        uv1 = distort_points(xyz, lut)
        xyz_recovered = undistort_points(uv1, lut)

        np.testing.assert_array_almost_equal(xyz, xyz_recovered, decimal=10)

    def test_round_trip_polynomial_with_coefficients(self) -> None:
        """Polynomial model with non-zero coefficients should round-trip."""
        coeffs = DistortionCoeffs(k2=0.02, k3=-0.01, k4=0.003)
        lut = build_distortion_lut(LensModel.POLYNOMIAL, coeffs=coeffs)
        rng = np.random.default_rng(456)
        xyz = rng.normal(size=(50, 3))
        xyz[:, 2] = np.abs(xyz[:, 2]) + 0.5
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        uv1 = distort_points(xyz, lut)
        xyz_recovered = undistort_points(uv1, lut)

        np.testing.assert_array_almost_equal(xyz, xyz_recovered, decimal=4)


# ---------------------------------------------------------------------------
# Full pipeline: pixel ↔ spherical ray with camera matrix K
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test spherical_rays_to_pixel_coords and pixel_coords_to_spherical_rays."""

    @pytest.fixture()
    def k_matrix(self) -> npt.NDArray[np.float64]:
        """Standard camera matrix: f=500, principal point at (320, 240)."""
        return np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def test_on_axis_projects_to_principal_point(self, k_matrix: npt.NDArray[np.float64]) -> None:
        """A ray along +Z should land at the principal point (cx, cy)."""
        lut = build_distortion_lut(LensModel.EQUIDISTANT)
        xyz = np.array([[0.0, 0.0, 1.0]])
        pixels = spherical_rays_to_pixel_coords(xyz, lut, k_matrix)
        assert pixels[0, 0] == pytest.approx(320.0, abs=1e-6)
        assert pixels[0, 1] == pytest.approx(240.0, abs=1e-6)

    @pytest.mark.parametrize("model", FISHEYE_MODELS)
    def test_full_round_trip(self, model: LensModel, k_matrix: npt.NDArray[np.float64]) -> None:
        """rays → pixels → rays should recover the original directions."""
        lut = build_distortion_lut(model)
        rng = np.random.default_rng(789)
        xyz = rng.normal(size=(30, 3))
        xyz[:, 2] = np.abs(xyz[:, 2]) + 0.5
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        pixels = spherical_rays_to_pixel_coords(xyz, lut, k_matrix)
        xyz_recovered = pixel_coords_to_spherical_rays(pixels, lut, k_matrix)

        np.testing.assert_array_almost_equal(xyz, xyz_recovered, decimal=3)
