"""Tests for scene camera management."""

from __future__ import annotations

import math

import numpy as np

from conic_sections.core.scene_camera import SceneCamera, SceneCameraManager


class TestSceneCamera:
    """SceneCamera transform and property tests."""

    def test_default_position(self) -> None:
        """Default position should be [3, 2, 3]."""
        cam = SceneCamera()
        np.testing.assert_allclose(cam.position, [3.0, 2.0, 3.0])

    def test_model_matrix_identity_at_origin(self) -> None:
        """Model matrix should be identity when position=0, rotation=0, scale=1."""
        cam = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            scale=1.0,
        )
        np.testing.assert_allclose(cam.model_matrix, np.eye(4), atol=1e-12)

    def test_model_matrix_translation(self) -> None:
        """Model matrix should encode the position as translation."""
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        cam = SceneCamera(position=pos)
        model = cam.model_matrix
        np.testing.assert_allclose(model[:3, 3], pos, atol=1e-12)

    def test_model_matrix_scale(self) -> None:
        """Scale should affect the model matrix diagonal."""
        cam = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            scale=2.0,
        )
        model = cam.model_matrix
        # The 3x3 submatrix should have determinant = scale^3
        det = float(np.linalg.det(model[:3, :3]))
        np.testing.assert_allclose(det, 8.0, atol=1e-10)

    def test_view_matrix_inverse_of_model(self) -> None:
        """View matrix times pose should give XZ-flip (det=+1, no mirror)."""
        cam = SceneCamera(
            position=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            rotation_euler=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            scale=1.0,
        )
        view = cam.view_matrix
        # Build pose matrix (no scale)
        rot = cam.rotation_matrix
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot
        pose[:3, 3] = cam.position
        product = view @ pose
        # Rows 0 and 2 are negated (X-flip + Z-flip), det stays +1
        expected = np.eye(4, dtype=np.float64)
        expected[0, 0] = -1.0
        expected[2, 2] = -1.0
        np.testing.assert_allclose(product, expected, atol=1e-10)

    def test_forward_direction_default(self) -> None:
        """Default camera should look along +Z."""
        cam = SceneCamera(rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(cam.forward, [0.0, 0.0, 1.0], atol=1e-12)

    def test_up_direction_default(self) -> None:
        """Default camera should have +Y up."""
        cam = SceneCamera(rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(cam.up, [0.0, 1.0, 0.0], atol=1e-12)

    def test_look_at_origin(self) -> None:
        """After look_at(origin), forward should point toward origin."""
        cam = SceneCamera(
            position=np.array([5.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cam.look_at(target)

        # Forward should point from [5,0,0] toward [0,0,0] = [-1,0,0]
        expected = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(cam.forward, expected, atol=1e-6)

    def test_set_intrinsics(self) -> None:
        """set_intrinsics should update the camera intrinsic parameters."""
        cam = SceneCamera()
        cam.set_intrinsics(fx=1000.0, fy=1000.0)
        assert cam.camera.intrinsics.fx == 1000.0
        assert cam.camera.intrinsics.fy == 1000.0

    def test_projection_matrix_shape(self) -> None:
        """Projection matrix should be 4x4."""
        cam = SceneCamera()
        proj = cam.projection_matrix
        assert proj.shape == (4, 4)


class TestSceneCameraFPSControls:
    """Tests for FPS-style scene camera controls."""

    def test_move_forward(self) -> None:
        """Moving forward should translate along +Z."""
        cam = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        cam.move(forward=1.0)
        # Default forward is +Z, so position.z should increase
        assert cam.position[2] > 0.0

    def test_move_backward(self) -> None:
        """Moving backward should translate along -Z."""
        cam = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        cam.move(forward=-1.0)
        assert cam.position[2] < 0.0

    def test_strafe_right(self) -> None:
        """Strafing right should translate along +X."""
        cam = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        cam.move(strafe=1.0)
        assert cam.position[0] > 0.0

    def test_move_fast_doubles_speed(self) -> None:
        """Fast flag should double the movement distance."""
        cam_normal = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        cam_fast = SceneCamera(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        cam_normal.move(forward=1.0, fast=False)
        cam_fast.move(forward=1.0, fast=True)
        np.testing.assert_allclose(cam_fast.position[2], 2.0 * cam_normal.position[2])

    def test_scroll_zoom(self) -> None:
        """Scroll should change FOV."""
        cam = SceneCamera(fov_y=60.0)
        cam.on_scroll(0.0, 1.0)
        assert cam.fov_y < 60.0  # scroll up = zoom in = decrease FOV

    def test_fov_clamp(self) -> None:
        """FOV should be clamped between 10 and 120 degrees."""
        cam = SceneCamera(fov_y=60.0)
        cam.on_scroll(0.0, 100.0)  # large zoom in
        assert cam.fov_y >= 10.0
        cam.on_scroll(0.0, -200.0)  # large zoom out
        assert cam.fov_y <= 120.0

    def test_reset_horizon(self) -> None:
        """reset_horizon should set roll to zero."""
        cam = SceneCamera(rotation_euler=np.array([0.1, 0.2, 0.5], dtype=np.float64))
        cam.reset_horizon()
        assert cam.rotation_euler[2] == 0.0

    def test_sync_cursor_no_rotation(self) -> None:
        """sync_cursor should update stored position without rotating."""
        cam = SceneCamera(rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        cam.sync_cursor(100.0, 200.0)
        euler_before = cam.rotation_euler.copy()
        cam.sync_cursor(150.0, 250.0)
        np.testing.assert_allclose(cam.rotation_euler, euler_before)

    def test_pitch_clamp(self) -> None:
        """Pitch should be clamped to avoid gimbal lock."""
        cam = SceneCamera(rotation_euler=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        cam.sync_cursor(0.0, 0.0)
        # Simulate large downward mouse movement (increases pitch)
        cam.on_cursor_pos(0.0, 10000.0)
        assert cam.rotation_euler[0] < math.pi / 2


class TestSceneCameraManager:
    """SceneCameraManager lifecycle tests."""

    def test_starts_empty(self) -> None:
        """Manager should start with no cameras."""
        mgr = SceneCameraManager()
        assert len(mgr.cameras) == 0
        assert mgr.active_index == -1
        assert mgr.active_camera is None

    def test_add_camera(self) -> None:
        """Adding a camera should increase count and set it active."""
        mgr = SceneCameraManager()
        cam = mgr.add_camera(name="Test Cam")
        assert len(mgr.cameras) == 1
        assert mgr.active_index == 0
        assert mgr.active_camera is cam
        assert cam.name == "Test Cam"

    def test_add_multiple_cameras(self) -> None:
        """Last added camera should become active."""
        mgr = SceneCameraManager()
        mgr.add_camera(name="Cam 1")
        cam2 = mgr.add_camera(name="Cam 2")
        assert mgr.active_index == 1
        assert mgr.active_camera is cam2

    def test_remove_camera(self) -> None:
        """Removing a camera should update the list and active index."""
        mgr = SceneCameraManager()
        mgr.add_camera(name="Cam 1")
        mgr.add_camera(name="Cam 2")
        mgr.remove_camera(0)
        assert len(mgr.cameras) == 1
        assert mgr.cameras[0].name == "Cam 2"

    def test_remove_last_camera(self) -> None:
        """Removing the only camera should reset active index."""
        mgr = SceneCameraManager()
        mgr.add_camera()
        mgr.remove_camera(0)
        assert len(mgr.cameras) == 0
        assert mgr.active_index == -1
        assert mgr.viewing_through is False

    def test_remove_invalid_index(self) -> None:
        """Removing an invalid index should raise IndexError."""
        mgr = SceneCameraManager()
        try:
            mgr.remove_camera(0)
            assert False, "Should have raised IndexError"  # noqa: B011
        except IndexError:
            pass

    def test_select_camera(self) -> None:
        """select_camera should update active_index."""
        mgr = SceneCameraManager()
        mgr.add_camera(name="Cam 1")
        mgr.add_camera(name="Cam 2")
        mgr.select_camera(0)
        assert mgr.active_index == 0

    def test_select_invalid_index(self) -> None:
        """Selecting an invalid index should raise IndexError."""
        mgr = SceneCameraManager()
        try:
            mgr.select_camera(0)
            assert False, "Should have raised IndexError"  # noqa: B011
        except IndexError:
            pass

    def test_toggle_viewport(self) -> None:
        """toggle_viewport should flip viewing_through."""
        mgr = SceneCameraManager()
        mgr.add_camera()
        assert mgr.viewing_through is False
        mgr.toggle_viewport()
        assert mgr.viewing_through is True
        mgr.toggle_viewport()
        assert mgr.viewing_through is False

    def test_toggle_viewport_no_camera(self) -> None:
        """toggle_viewport with no camera should do nothing."""
        mgr = SceneCameraManager()
        mgr.toggle_viewport()
        assert mgr.viewing_through is False

    def test_cycle_camera(self) -> None:
        """cycle_camera should advance to the next camera."""
        mgr = SceneCameraManager()
        mgr.add_camera(name="Cam 1")
        mgr.add_camera(name="Cam 2")
        mgr.add_camera(name="Cam 3")
        mgr.select_camera(0)
        mgr.cycle_camera()
        assert mgr.active_index == 1
        mgr.cycle_camera()
        assert mgr.active_index == 2
        mgr.cycle_camera()
        assert mgr.active_index == 0  # wraps around

    def test_cycle_empty(self) -> None:
        """cycle_camera with no cameras should do nothing."""
        mgr = SceneCameraManager()
        mgr.cycle_camera()
        assert mgr.active_index == -1

    def test_get_view_projection_orbit(self) -> None:
        """When not viewing through, should return orbit matrices."""
        mgr = SceneCameraManager()
        orbit_view = np.eye(4, dtype=np.float64)
        orbit_proj = np.eye(4, dtype=np.float64)
        view, proj, eye = mgr.get_view_projection(orbit_view, orbit_proj)
        np.testing.assert_allclose(view, orbit_view)
        np.testing.assert_allclose(proj, orbit_proj)

    def test_get_view_projection_scene(self) -> None:
        """When viewing through a scene camera, should return its matrices."""
        mgr = SceneCameraManager()
        cam = mgr.add_camera(position=np.array([1.0, 2.0, 3.0], dtype=np.float64))
        mgr.toggle_viewport()

        orbit_view = np.eye(4, dtype=np.float64)
        orbit_proj = np.eye(4, dtype=np.float64)
        view, proj, eye = mgr.get_view_projection(orbit_view, orbit_proj)

        # Should be the scene camera's matrices, not orbit
        np.testing.assert_allclose(view, cam.view_matrix)
        np.testing.assert_allclose(eye, [1.0, 2.0, 3.0])

    def test_add_camera_with_look_at(self) -> None:
        """add_camera with look_at_target should orient the camera."""
        mgr = SceneCameraManager()
        cam = mgr.add_camera(
            position=np.array([5.0, 0.0, 0.0], dtype=np.float64),
            look_at_target=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        expected_forward = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_allclose(cam.forward, expected_forward, atol=1e-6)

    def test_auto_naming(self) -> None:
        """Cameras should get auto-generated names when none provided."""
        mgr = SceneCameraManager()
        cam1 = mgr.add_camera()
        cam2 = mgr.add_camera()
        assert cam1.name == "Camera 1"
        assert cam2.name == "Camera 2"
