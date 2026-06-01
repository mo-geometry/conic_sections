"""Scene camera management — place and control cameras in the 3D world.

Coordinate convention (consistent across the repository):
- The camera looks along its local **+Z** axis.
- **+Y** is up, **+X** is right.
- The camera mesh has its lens on the +Z face, so the mesh, frustum,
  and view direction are all aligned.

Euler angles use the ZYX intrinsic convention via ``euler_to_rotation``:
- ``rotation_euler[0]`` (roll param) → pitch (look up/down, rotation about X)
- ``rotation_euler[1]`` (pitch param) → yaw (look left/right, rotation about Y)
- ``rotation_euler[2]`` (yaw param) → roll (tilt, rotation about Z)

FPS-style controls when viewing through a scene camera:
- **Mouse movement** → yaw / pitch (look around)
- **Right-click + drag** → roll (tilt)
- **Scroll** → zoom (change FOV)
- **W / Up** → move forward (+Z), **S / Down** → move backward
- **A / Left** → strafe left, **D / Right** → strafe right
- **Shift** modifier → double movement speed
- **Double right-click** → reset roll to zero (horizon level)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from conic_sections.core.camera import Camera, Extrinsics, Intrinsics
from conic_sections.core.transforms import euler_to_rotation


@dataclass
class SceneCamera:
    """A camera placed in the 3D scene.

    The camera looks along its local +Z axis.  Its mesh, frustum, and
    view direction all share this convention.

    Attributes:
        name: Human-readable label for UI display.
        camera: The underlying Camera model (intrinsics + extrinsics).
        position: World-space position [x, y, z].
        rotation_euler: Euler angles [pitch, yaw, roll] in radians.
            pitch = look up/down (X rot), yaw = look left/right (Y rot),
            roll = tilt (Z rot).
        scale: Uniform scale factor for the camera mesh (visual only).
        fov_y: Vertical field of view in degrees (for zoom control).
        visible: Whether to render this camera's mesh in the scene.
        show_frustum: Whether to render the view frustum wireframe.
    """

    name: str = "Camera 1"
    camera: Camera = field(default_factory=Camera)
    position: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([3.0, 2.0, 3.0], dtype=np.float64)
    )
    rotation_euler: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    scale: float = 1.0
    fov_y: float = 60.0
    aspect_ratio: float = 16.0 / 9.0
    visible: bool = True
    show_frustum: bool = True
    _frustum_dirty: bool = False

    # ------------------------------------------------------------------
    # Rotation and transform properties
    # ------------------------------------------------------------------

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Compute the 3x3 rotation matrix from Euler angles."""
        return euler_to_rotation(
            float(self.rotation_euler[0]),  # pitch → roll param (X)
            float(self.rotation_euler[1]),  # yaw   → pitch param (Y)
            float(self.rotation_euler[2]),  # roll  → yaw param (Z)
        )

    @property
    def model_matrix(self) -> npt.NDArray[np.float64]:
        """4x4 model matrix for rendering the camera mesh in world space."""
        mat = np.eye(4, dtype=np.float64)

        # Scale
        mat[0, 0] = self.scale
        mat[1, 1] = self.scale
        mat[2, 2] = self.scale

        # Rotation
        rot = self.rotation_matrix
        rot_4x4 = np.eye(4, dtype=np.float64)
        rot_4x4[:3, :3] = rot
        mat = rot_4x4 @ mat

        # Translation
        mat[0, 3] = self.position[0]
        mat[1, 3] = self.position[1]
        mat[2, 3] = self.position[2]

        return mat

    @property
    def view_matrix(self) -> npt.NDArray[np.float64]:
        """4x4 view matrix for rendering from this camera's perspective.

        OpenGL expects the camera to look along -Z, but our camera looks
        along +Z.  We negate both the X row and Z row of the pose-inverse
        so that:

        - Row 0 becomes ``-right`` (matching ``forward x up``, the OpenGL
          side vector convention used by ``glm::lookAt``).
        - Row 2 becomes ``-forward`` (mapping our +Z to OpenGL's -Z).

        Negating two rows preserves the determinant (+1 → +1), so the
        winding order of triangles is unchanged and no ``front_face``
        workaround is needed.
        """
        rot = self.rotation_matrix

        # Standard pose inverse: V = [R^T | -R^T * p]
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = rot.T
        view[:3, 3] = -rot.T @ self.position

        # Negate rows 0 and 2 to match glm::lookAt convention (det stays +1)
        view[0, :] = -view[0, :]
        view[2, :] = -view[2, :]

        return view

    @property
    def projection_matrix(self) -> npt.NDArray[np.float64]:
        """4x4 perspective projection matrix from fov_y and aspect ratio."""
        aspect = self.camera.width / self.camera.height
        f = 1.0 / math.tan(math.radians(self.fov_y) / 2.0)
        n, fa = self.camera.near, self.camera.far

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (fa + n) / (n - fa)
        proj[2, 3] = (2.0 * fa * n) / (n - fa)
        proj[3, 2] = -1.0
        return proj

    # ------------------------------------------------------------------
    # Direction vectors (world space)
    # ------------------------------------------------------------------

    @property
    def forward(self) -> npt.NDArray[np.float64]:
        """Unit vector along the camera's look direction (local +Z)."""
        rot = self.rotation_matrix
        fwd: npt.NDArray[np.float64] = rot[:, 2]
        return fwd

    @property
    def up(self) -> npt.NDArray[np.float64]:
        """Unit vector in the camera's up direction (local +Y)."""
        rot = self.rotation_matrix
        u: npt.NDArray[np.float64] = rot[:, 1]
        return u

    @property
    def right(self) -> npt.NDArray[np.float64]:
        """Unit vector in the camera's right direction (local +X)."""
        rot = self.rotation_matrix
        r: npt.NDArray[np.float64] = rot[:, 0]
        return r

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    def look_at(self, target: npt.NDArray[np.float64]) -> None:
        """Orient the camera so its +Z axis points toward *target*.

        Args:
            target: World-space point to look at [x, y, z].
        """
        d = target - self.position
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            return
        d = d / norm

        # Yaw: horizontal angle from +Z toward +X
        yaw = float(np.arctan2(d[0], d[2]))

        # Pitch: elevation angle
        horiz = float(np.sqrt(d[0] ** 2 + d[2] ** 2))
        pitch = float(np.arctan2(-d[1], horiz))

        self.rotation_euler = np.array([pitch, yaw, 0.0], dtype=np.float64)

    def reset_horizon(self) -> None:
        """Reset roll to zero (level the horizon)."""
        self.rotation_euler[2] = 0.0

    def set_intrinsics(
        self,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
    ) -> None:
        """Update camera intrinsic parameters."""
        if fx is not None:
            self.camera.intrinsics.fx = fx
        if fy is not None:
            self.camera.intrinsics.fy = fy
        if cx is not None:
            self.camera.intrinsics.cx = cx
        if cy is not None:
            self.camera.intrinsics.cy = cy

    def set_camera_format(self, fov_y: float, aspect_ratio: float) -> None:
        """Update the camera format (FOV and aspect ratio).

        This marks the frustum as dirty so it will be regenerated if displayed.

        Args:
            fov_y: Vertical field of view in degrees.
            aspect_ratio: Width / height aspect ratio.
        """
        self.fov_y = fov_y
        self.aspect_ratio = aspect_ratio
        self._frustum_dirty = True

    def get_frustum_geometry(
        self,
        near: float = 0.3,
        far: float = 2.0,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Generate frustum line geometry for this camera's current FOV/aspect.

        Args:
            near: Near plane distance.
            far: Far plane distance.

        Returns:
            Tuple of (positions, colors) arrays for GL_LINES rendering.
        """
        from conic_sections.assets.camera_mesh import create_frustum_lines

        return create_frustum_lines(
            fov_y=self.fov_y,
            aspect=self.aspect_ratio,
            near=near,
            far=far,
        )

    # ------------------------------------------------------------------
    # FPS-style controls (used when viewing through this camera)
    # ------------------------------------------------------------------

    _rolling: bool = False
    _last_x: float = 0.0
    _last_y: float = 0.0
    _last_click_time: float = 0.0
    _sensitivity: float = 0.003
    _roll_sensitivity: float = 0.004
    _move_speed: float = 0.15
    _zoom_speed: float = 2.0

    def on_mouse_button(self, button: int, action: int, _mods: int) -> None:
        """Handle mouse button for roll and double-click horizon reset.

        Right-click + drag = roll around Z axis.
        Double right-click = reset horizon.
        """
        if button == 1:  # Right mouse button
            if action == 1:  # Press
                now = time.monotonic()
                if now - self._last_click_time < 0.35:
                    self.reset_horizon()
                self._last_click_time = now
                self._rolling = True
            else:
                self._rolling = False

    def on_cursor_pos(self, x: float, y: float) -> None:
        """Handle cursor movement.

        Without right-click: yaw/pitch (look around).
        With right-click held: roll (tilt).
        """
        dx = x - self._last_x
        dy = y - self._last_y

        if self._rolling:
            # Right-click drag → roll around Z axis
            self.rotation_euler[2] -= dx * self._roll_sensitivity
        else:
            # Free mouse → yaw/pitch
            # Negate dx because the Z-flip in the view matrix inverts handedness.
            # This causes rotations to appear reversed, so we compensate here.
            self.rotation_euler[1] -= dx * self._sensitivity
            self.rotation_euler[0] += dy * self._sensitivity
            # Clamp pitch to avoid gimbal lock
            self.rotation_euler[0] = float(
                np.clip(
                    self.rotation_euler[0],
                    -math.pi / 2 + 0.01,
                    math.pi / 2 - 0.01,
                )
            )

        self._last_x = x
        self._last_y = y

    def on_scroll(self, _x_offset: float, y_offset: float) -> None:
        """Handle scroll for zoom (change FOV).

        Scroll up → zoom in (decrease FOV), scroll down → zoom out.
        """
        self.fov_y -= y_offset * self._zoom_speed
        self.fov_y = float(np.clip(self.fov_y, 10.0, 120.0))

    def move(
        self,
        forward: float = 0.0,
        strafe: float = 0.0,
        vertical: float = 0.0,
        fast: bool = False,
    ) -> None:
        """Translate the camera on the horizontal plane, or vertically.

        Forward/strafe movement is projected onto the XZ plane so the
        camera stays at a constant height regardless of pitch.  Use
        *vertical* for explicit up/down movement.

        Args:
            forward: Movement along the camera's horizontal forward direction.
            strafe: Movement along the camera's horizontal right direction.
            vertical: Movement along the world +Y axis (positive = up).
            fast: If True, double the movement speed.
        """
        speed = self._move_speed * (0.25 if fast else 0.1)

        # Project forward and right onto the XZ plane for horizontal movement
        fwd_xz = self.forward.copy()
        fwd_xz[1] = 0.0
        fwd_norm = float(np.linalg.norm(fwd_xz))
        if fwd_norm > 1e-8:
            fwd_xz /= fwd_norm

        right_xz = self.right.copy()
        right_xz[1] = 0.0
        right_norm = float(np.linalg.norm(right_xz))
        if right_norm > 1e-8:
            right_xz /= right_norm

        self.position = (
            self.position
            + fwd_xz * forward * speed
            + right_xz * strafe * speed
            + np.array([0.0, vertical * speed, 0.0], dtype=np.float64)
        )

    def sync_cursor(self, x: float, y: float) -> None:
        """Update stored cursor position without rotating.

        Call this when the mouse is over ImGui or orbit camera is active,
        to prevent jumps when re-entering first-person mode.
        """
        self._last_x = x
        self._last_y = y


class SceneCameraManager:
    """Manages multiple scene cameras and viewport switching.

    Attributes:
        cameras: List of scene cameras.
        active_index: Index of the currently selected scene camera (-1 for none).
        viewing_through: Whether the viewport shows a scene camera's view.
        show_world_axes: Whether to render world coordinate axes.
    """

    def __init__(self) -> None:
        self.cameras: list[SceneCamera] = []
        self.active_index: int = -1
        self.viewing_through: bool = False
        self.show_world_axes: bool = True

    @property
    def active_camera(self) -> SceneCamera | None:
        """Return the currently selected scene camera, or None."""
        if 0 <= self.active_index < len(self.cameras):
            return self.cameras[self.active_index]
        return None

    def add_camera(
        self,
        name: str | None = None,
        position: npt.NDArray[np.float64] | None = None,
        look_at_target: npt.NDArray[np.float64] | None = None,
    ) -> SceneCamera:
        """Add a new scene camera.

        Args:
            name: Display name. Auto-generated if None.
            position: World-space position. Default placement if None.
            look_at_target: Optional target point to orient toward.

        Returns:
            The newly created SceneCamera.
        """
        idx = len(self.cameras) + 1
        cam_name = name or f"Camera {idx}"

        cam = SceneCamera(
            name=cam_name,
            camera=Camera(
                intrinsics=Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0),
                extrinsics=Extrinsics(),
            ),
        )

        if position is not None:
            cam.position = position.copy()

        if look_at_target is not None:
            cam.look_at(look_at_target)

        self.cameras.append(cam)
        self.active_index = len(self.cameras) - 1
        return cam

    def remove_camera(self, index: int) -> None:
        """Remove a camera by index."""
        if not 0 <= index < len(self.cameras):
            msg = f"Camera index {index} out of range [0, {len(self.cameras)})"
            raise IndexError(msg)

        self.cameras.pop(index)

        if len(self.cameras) == 0:
            self.active_index = -1
            self.viewing_through = False
        elif self.active_index >= len(self.cameras):
            self.active_index = len(self.cameras) - 1
        elif self.active_index > index:
            self.active_index -= 1

    def select_camera(self, index: int) -> None:
        """Select a camera by index."""
        if not 0 <= index < len(self.cameras):
            msg = f"Camera index {index} out of range [0, {len(self.cameras)})"
            raise IndexError(msg)
        self.active_index = index

    def toggle_viewport(self) -> None:
        """Toggle between orbit camera and the active scene camera view."""
        if self.active_camera is not None:
            self.viewing_through = not self.viewing_through

    def cycle_camera(self) -> None:
        """Cycle to the next scene camera."""
        if len(self.cameras) == 0:
            return
        self.active_index = (self.active_index + 1) % len(self.cameras)

    def get_view_projection(
        self,
        orbit_view: npt.NDArray[np.float64],
        orbit_proj: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the current viewport's view and projection matrices.

        Args:
            orbit_view: The orbit camera's 4x4 view matrix.
            orbit_proj: The orbit camera's 4x4 projection matrix.

        Returns:
            Tuple of (view_matrix, projection_matrix, eye_position).
        """
        if self.viewing_through and self.active_camera is not None:
            cam = self.active_camera
            view = cam.view_matrix
            proj = cam.projection_matrix
            eye = cam.position.copy()
            return view, proj, eye

        # Extract eye position from orbit view matrix
        r_inv = orbit_view[:3, :3].T
        eye = -r_inv @ orbit_view[:3, 3]
        return orbit_view, orbit_proj, eye
