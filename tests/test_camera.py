"""Tests for camera intrinsics, extrinsics, and projection matrices."""

from __future__ import annotations

import numpy as np
import pytest

from conic_sections.core.camera import Camera, Extrinsics, Intrinsics


class TestIntrinsics:
    """Intrinsic parameter and matrix tests."""

    def test_default_matrix_shape(self) -> None:
        k = Intrinsics()
        assert k.matrix.shape == (3, 3)

    def test_matrix_diagonal(self) -> None:
        intr = Intrinsics(fx=500.0, fy=600.0)
        k = intr.matrix
        assert k[0, 0] == pytest.approx(500.0)
        assert k[1, 1] == pytest.approx(600.0)
        assert k[2, 2] == pytest.approx(1.0)

    def test_principal_point(self) -> None:
        intr = Intrinsics(cx=100.0, cy=200.0)
        k = intr.matrix
        assert k[0, 2] == pytest.approx(100.0)
        assert k[1, 2] == pytest.approx(200.0)

    def test_default_distortion_is_zero(self) -> None:
        intr = Intrinsics()
        np.testing.assert_array_equal(intr.dist_coeffs, np.zeros(5))


class TestExtrinsics:
    """Extrinsic pose and view-matrix tests."""

    def test_identity_view_matrix(self) -> None:
        ext = Extrinsics()
        np.testing.assert_array_almost_equal(ext.view_matrix, np.eye(4))

    def test_translation_only(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        ext = Extrinsics(position=pos)
        view = ext.view_matrix
        # With identity rotation, translation column = -position
        np.testing.assert_array_almost_equal(view[:3, 3], -pos)

    def test_view_matrix_shape(self) -> None:
        ext = Extrinsics()
        assert ext.view_matrix.shape == (4, 4)


class TestCamera:
    """Full camera model tests."""

    def test_projection_matrix_shape(self, default_camera: Camera) -> None:
        assert default_camera.projection_matrix.shape == (4, 4)

    def test_projection_bottom_row(self, default_camera: Camera) -> None:
        proj = default_camera.projection_matrix
        # OpenGL-style: last row should be [0, 0, -1, 0]
        np.testing.assert_array_almost_equal(proj[3], [0, 0, -1, 0])

    def test_near_far_affects_projection(self) -> None:
        cam_a = Camera(near=0.1, far=100.0)
        cam_b = Camera(near=0.5, far=50.0)
        # The [2,3] element encodes -2*f*n/(f-n) — should differ for different near/far
        assert cam_a.projection_matrix[2, 3] != pytest.approx(cam_b.projection_matrix[2, 3])

    def test_view_matrix_delegates_to_extrinsics(self, translated_camera: Camera) -> None:
        np.testing.assert_array_almost_equal(
            translated_camera.view_matrix,
            translated_camera.extrinsics.view_matrix,
        )
