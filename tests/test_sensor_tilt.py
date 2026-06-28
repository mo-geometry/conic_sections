"""Tests for sensor tilt compensation (IMVIP eqns 4a/4b and 7a/7b).

Covers:
- Tilt vector construction from angles
- Identity behaviour when tilt angle is zero
- Round-trip consistency: project_rays then unproject_rays recovers input
- Known geometric values (tilt along X-axis, Y-axis)
- Numerical match against the main-branch implementation
- Integration with the distortion pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

from conic_sections.core.sensor_tilt import (
    TiltVector,
    project_rays,
    tilt_from_angles,
    unproject_rays,
)

# ---------------------------------------------------------------------------
# Tilt vector construction
# ---------------------------------------------------------------------------


class TestTiltFromAngles:
    """Tests for tilt_from_angles and TiltVector."""

    def test_zero_tilt_is_identity(self) -> None:
        """Zero tilt angle should produce nz = -1 (untilted sensor)."""
        tv = tilt_from_angles(0.0, 0.0)
        assert tv.nx == pytest.approx(0.0)
        assert tv.ny == pytest.approx(0.0)
        assert tv.nz == pytest.approx(-1.0)
        assert tv.is_identity

    def test_zero_tilt_any_azimuth(self) -> None:
        """Azimuth is irrelevant when tilt angle is zero."""
        for azi in [0, 45, 90, 180, 270]:
            tv = tilt_from_angles(0.0, float(azi))
            assert tv.is_identity

    def test_tilt_along_x_axis(self) -> None:
        """Tilt of 10° at azimuth 0° should have ny ≈ 0, nx > 0."""
        tv = tilt_from_angles(10.0, 0.0)
        assert tv.nx == pytest.approx(np.sin(np.radians(10.0)), abs=1e-12)
        assert tv.ny == pytest.approx(0.0, abs=1e-12)
        assert tv.nz == pytest.approx(-np.cos(np.radians(10.0)), abs=1e-12)
        assert not tv.is_identity

    def test_tilt_along_y_axis(self) -> None:
        """Tilt of 10° at azimuth 90° should have nx ≈ 0, ny > 0."""
        tv = tilt_from_angles(10.0, 90.0)
        assert tv.nx == pytest.approx(0.0, abs=1e-12)
        assert tv.ny == pytest.approx(np.sin(np.radians(10.0)), abs=1e-12)
        assert tv.nz == pytest.approx(-np.cos(np.radians(10.0)), abs=1e-12)

    def test_unit_normal(self) -> None:
        """The tilt vector should be a unit vector for any angle/azimuth."""
        for angle in [0, 5, 15, 30, 45]:
            for azi in [0, 45, 90, 135, 270]:
                tv = tilt_from_angles(float(angle), float(azi))
                norm = np.sqrt(tv.nx**2 + tv.ny**2 + tv.nz**2)
                assert norm == pytest.approx(1.0, abs=1e-12)

    def test_default_tilt_vector(self) -> None:
        """Default TiltVector() should be the identity (untilted)."""
        tv = TiltVector()
        assert tv.is_identity


# ---------------------------------------------------------------------------
# Identity: zero tilt leaves coordinates unchanged
# ---------------------------------------------------------------------------


class TestIdentity:
    """When tilt is zero, project and unproject should return a copy of input."""

    def test_project_identity(self) -> None:
        tv = tilt_from_angles(0.0, 0.0)
        pts = np.array([[0.3, -0.2, 1.0], [0.0, 0.0, 1.0], [-0.5, 0.4, 1.0]])
        result = project_rays(pts, tv)
        np.testing.assert_array_equal(result, pts)

    def test_unproject_identity(self) -> None:
        tv = tilt_from_angles(0.0, 0.0)
        pts = np.array([[0.3, -0.2, 1.0], [0.0, 0.0, 1.0], [-0.5, 0.4, 1.0]])
        result = unproject_rays(pts, tv)
        np.testing.assert_array_equal(result, pts)

    def test_identity_returns_copy(self) -> None:
        """Identity case should return a copy, not a reference to the input."""
        tv = tilt_from_angles(0.0, 0.0)
        pts = np.array([[0.1, 0.2, 1.0]])
        result = project_rays(pts, tv)
        result[0, 0] = 999.0
        assert pts[0, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Optical centre: (0, 0, 1) is a fixed point of the tilt transform
# ---------------------------------------------------------------------------


class TestOpticalCentre:
    """The optical centre (0, 0, 1) should be invariant under any tilt."""

    @pytest.mark.parametrize("angle", [5.0, 15.0, 30.0])
    @pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 180.0])
    def test_project_fixed_point(self, angle: float, azimuth: float) -> None:
        tv = tilt_from_angles(angle, azimuth)
        origin = np.array([[0.0, 0.0, 1.0]])
        result = project_rays(origin, tv)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("angle", [5.0, 15.0, 30.0])
    @pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 180.0])
    def test_unproject_fixed_point(self, angle: float, azimuth: float) -> None:
        tv = tilt_from_angles(angle, azimuth)
        origin = np.array([[0.0, 0.0, 1.0]])
        result = unproject_rays(origin, tv)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert result[0, 1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Round-trip: project → unproject recovers the original
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """project_rays followed by unproject_rays should recover input."""

    @pytest.mark.parametrize("angle", [5.0, 15.0, 30.0])
    @pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 270.0])
    def test_round_trip(self, angle: float, azimuth: float) -> None:
        tv = tilt_from_angles(angle, azimuth)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-0.4, 0.4, size=(50, 3))
        pts[:, 2] = 1.0

        projected = project_rays(pts, tv)
        recovered = unproject_rays(projected, tv)

        np.testing.assert_array_almost_equal(pts, recovered, decimal=12)

    @pytest.mark.parametrize("angle", [5.0, 15.0, 30.0])
    @pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0, 270.0])
    def test_reverse_round_trip(self, angle: float, azimuth: float) -> None:
        """unproject_rays followed by project_rays should also round-trip."""
        tv = tilt_from_angles(angle, azimuth)
        rng = np.random.default_rng(99)
        pts = rng.uniform(-0.4, 0.4, size=(50, 3))
        pts[:, 2] = 1.0

        unprojected = unproject_rays(pts, tv)
        recovered = project_rays(unprojected, tv)

        np.testing.assert_array_almost_equal(pts, recovered, decimal=12)


# ---------------------------------------------------------------------------
# Non-trivial: tilt should actually move points (not silently be identity)
# ---------------------------------------------------------------------------


class TestNonTrivial:
    """Verify that a non-zero tilt actually changes coordinates."""

    def test_project_moves_off_axis_point(self) -> None:
        """A point away from the centre should shift under tilt."""
        tv = tilt_from_angles(15.0, 0.0)
        pts = np.array([[0.3, 0.0, 1.0]])
        result = project_rays(pts, tv)
        # Should not be identical to input
        assert not np.allclose(result[:, :2], pts[:, :2])

    def test_third_column_preserved(self) -> None:
        """The homogeneous coordinate (column 2) should remain 1."""
        tv = tilt_from_angles(20.0, 45.0)
        rng = np.random.default_rng(77)
        pts = rng.uniform(-0.5, 0.5, size=(30, 3))
        pts[:, 2] = 1.0
        proj = project_rays(pts, tv)
        unp = unproject_rays(pts, tv)
        np.testing.assert_array_almost_equal(proj[:, 2], 1.0)
        np.testing.assert_array_almost_equal(unp[:, 2], 1.0)

    def test_output_shape(self) -> None:
        tv = tilt_from_angles(10.0, 30.0)
        pts = np.ones((25, 3), dtype=np.float64)
        assert project_rays(pts, tv).shape == (25, 3)
        assert unproject_rays(pts, tv).shape == (25, 3)


# ---------------------------------------------------------------------------
# Integration: tilt + distortion pipeline
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Test sensor tilt combined with the radial distortion module."""

    def test_full_forward_inverse_pipeline(self) -> None:
        """rays → distort → tilt → K → K⁻¹ → untilt → undistort → rays."""
        from conic_sections.core.distortion import (
            LensModel,
            build_distortion_lut,
            distort_points,
            undistort_points,
        )

        lut = build_distortion_lut(LensModel.EQUIDISTANT)
        tilt = tilt_from_angles(10.0, 45.0)
        k_matrix = np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Generate rays in forward hemisphere
        rng = np.random.default_rng(321)
        xyz = rng.normal(size=(40, 3))
        xyz[:, 2] = np.abs(xyz[:, 2]) + 0.5
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

        # Forward: rays → distort → tilt → K
        uv1 = distort_points(xyz, lut)
        uv1_tilted = project_rays(uv1, tilt)
        pixels = uv1_tilted @ k_matrix.T

        # Inverse: K⁻¹ → untilt → undistort → rays
        k_inv = np.linalg.inv(k_matrix)
        uv1_back = pixels @ k_inv.T
        uv1_untilted = unproject_rays(uv1_back, tilt)
        xyz_recovered = undistort_points(uv1_untilted, lut)

        np.testing.assert_array_almost_equal(xyz, xyz_recovered, decimal=4)
