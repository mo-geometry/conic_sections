"""Tests for the orbit camera controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conic_sections.core.orbit_camera import OrbitCamera


class TestOrbitCamera:
    """Orbit camera behaviour tests."""

    def test_default_position(self) -> None:
        cam = OrbitCamera()
        eye = cam.eye
        # Default: distance=5, azimuth=0, elevation=0.3
        # Should be off-axis, not at origin
        assert np.linalg.norm(eye) > 0

    def test_distance_from_target(self) -> None:
        cam = OrbitCamera(distance=7.0)
        dist = np.linalg.norm(cam.eye - cam.target)
        assert dist == pytest.approx(7.0, abs=1e-10)

    def test_view_matrix_shape(self) -> None:
        cam = OrbitCamera()
        assert cam.view_matrix.shape == (4, 4)

    def test_scroll_zooms_in(self) -> None:
        cam = OrbitCamera(distance=5.0)
        cam.on_scroll(0.0, 2.0)  # Scroll up = zoom in
        assert cam.distance < 5.0

    def test_scroll_zooms_out(self) -> None:
        cam = OrbitCamera(distance=5.0)
        cam.on_scroll(0.0, -2.0)  # Scroll down = zoom out
        assert cam.distance > 5.0

    def test_zoom_respects_limits(self) -> None:
        cam = OrbitCamera(distance=2.0)
        cam.min_distance = 1.0
        cam.max_distance = 10.0
        # Zoom way in
        for _ in range(100):
            cam.on_scroll(0.0, 5.0)
        assert cam.distance >= cam.min_distance
        # Zoom way out
        for _ in range(100):
            cam.on_scroll(0.0, -5.0)
        assert cam.distance <= cam.max_distance

    def test_drag_changes_azimuth(self) -> None:
        cam = OrbitCamera(azimuth=0.0)
        cam.on_mouse_button(0, 1, 0)  # Press left
        cam.on_cursor_pos(100.0, 100.0)  # Set initial position
        cam.on_cursor_pos(200.0, 100.0)  # Drag right
        cam.on_mouse_button(0, 0, 0)  # Release
        assert cam.azimuth != 0.0

    def test_elevation_clamped(self) -> None:
        cam = OrbitCamera(elevation=0.0)
        cam.on_mouse_button(0, 1, 0)  # Press
        cam.on_cursor_pos(100.0, 100.0)
        # Drag way up
        cam.on_cursor_pos(100.0, -10000.0)
        assert cam.elevation < math.pi / 2
        assert cam.elevation > -math.pi / 2

    def test_no_drag_when_not_pressed(self) -> None:
        cam = OrbitCamera(azimuth=0.0, elevation=0.0)
        cam.on_cursor_pos(100.0, 100.0)
        cam.on_cursor_pos(200.0, 200.0)
        assert cam.azimuth == 0.0
        assert cam.elevation == 0.0
