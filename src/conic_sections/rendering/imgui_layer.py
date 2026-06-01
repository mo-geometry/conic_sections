"""Dear ImGui integration layer for the ModernGL renderer.

Handles ImGui context lifecycle (init, new-frame, render, shutdown) and
provides the camera control panel for managing scene cameras.

The panel can be toggled on/off with Tab or middle-mouse-click.  When
visible, ImGui captures mouse input so the orbit camera stays still.
When hidden, all input goes to the orbit camera.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

if TYPE_CHECKING:
    from conic_sections.core.scene_camera import SceneCameraManager


class ImGuiLayer:
    """Manages Dear ImGui lifecycle and renders the camera control panel.

    The panel starts hidden.  Press **Tab** or **middle-click** to toggle
    it.  While visible, ImGui owns the mouse; while hidden, the orbit
    camera owns it.

    Usage::

        layer = ImGuiLayer(window)
        # In render loop:
        layer.new_frame()
        if layer.panel_visible:
            layer.render_camera_panel(camera_manager)
        layer.render_draw_data()
        # On shutdown:
        layer.shutdown()
    """

    def __init__(self, window: Any) -> None:
        """Initialise ImGui with a GLFW window.

        Args:
            window: GLFW window handle.
        """
        imgui.create_context()
        self._impl = GlfwRenderer(window, attach_callbacks=False)
        self._window = window
        self.panel_visible: bool = False

    # ------------------------------------------------------------------
    # GLFW event forwarding — call these from the GLFW callbacks so
    # that ImGui can track hover state and update want_capture_mouse.
    # ------------------------------------------------------------------

    def on_mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        """Forward a GLFW mouse-button event to ImGui."""
        self._impl.mouse_button_callback(window, button, action, mods)

    def on_cursor_pos(self, window: Any, x: float, y: float) -> None:
        """Forward a GLFW cursor-position event to ImGui.

        The backend reads the position from glfw directly, so the
        x/y args aren't used — but we must call the method so ImGui
        updates its hover state.
        """
        self._impl.mouse_callback(window, x, y)

    def on_scroll(self, window: Any, x_offset: float, y_offset: float) -> None:
        """Forward a GLFW scroll event to ImGui."""
        self._impl.scroll_callback(window, x_offset, y_offset)

    def on_key(
        self,
        window: Any,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        """Forward a GLFW key event to ImGui."""
        self._impl.keyboard_callback(window, key, scancode, action, mods)

    def on_char(self, window: Any, char: int) -> None:
        """Forward a GLFW character event to ImGui."""
        self._impl.char_callback(window, char)

    # ------------------------------------------------------------------
    # Query state
    # ------------------------------------------------------------------

    @property
    def want_capture_mouse(self) -> bool:
        """True if ImGui wants mouse input (panel visible and hovered)."""
        if not self.panel_visible:
            return False
        return bool(imgui.get_io().want_capture_mouse)

    @property
    def want_capture_keyboard(self) -> bool:
        """True if ImGui wants keyboard input (typing in a field)."""
        if not self.panel_visible:
            return False
        return bool(imgui.get_io().want_capture_keyboard)

    def toggle_panel(self) -> None:
        """Toggle the camera control panel on/off."""
        self.panel_visible = not self.panel_visible

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def new_frame(self) -> None:
        """Begin a new ImGui frame. Call once per render loop iteration."""
        self._impl.process_inputs()
        imgui.new_frame()

    def render_draw_data(self) -> None:
        """Finalise and render ImGui draw data. Call after all panels."""
        imgui.render()
        self._impl.render(imgui.get_draw_data())

    def shutdown(self) -> None:
        """Clean up ImGui resources."""
        self._impl.shutdown()

    # ------------------------------------------------------------------
    # Camera control panel
    # ------------------------------------------------------------------

    def render_camera_panel(self, manager: SceneCameraManager) -> None:
        """Render the camera management ImGui panel.

        Provides controls for:
        - Adding/removing scene cameras
        - Selecting the active camera
        - Adjusting position and orientation
        - Toggling viewport (orbit vs scene camera)
        - Toggling frustum visibility

        Args:
            manager: The SceneCameraManager to control.
        """
        if not self.panel_visible:
            return

        imgui.set_next_window_pos((10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((320, 0), imgui.Cond_.first_use_ever)

        expanded, _ = imgui.begin("Camera Controls", True)
        if not expanded:
            imgui.end()
            return

        # --- Viewport mode indicator ---
        if manager.viewing_through and manager.active_camera is not None:
            imgui.text_colored((0.3, 1.0, 0.3, 1.0), f"Viewing: {manager.active_camera.name}")
        else:
            imgui.text_colored((0.7, 0.7, 1.0, 1.0), "Viewing: Orbit Camera")

        imgui.separator()

        # --- Add / toggle buttons ---
        if imgui.button("Add Camera"):
            n = len(manager.cameras) + 1
            import math

            angle = (n - 1) * math.pi / 3.0
            pos_x = 4.0 * math.sin(angle)
            pos_z = 4.0 * math.cos(angle)
            import numpy as np

            cam = manager.add_camera(
                position=np.array([pos_x, 2.5, pos_z], dtype=np.float64),
            )
            cam.look_at(np.array([0.0, 1.0, 0.0], dtype=np.float64))

        imgui.same_line()

        if manager.active_camera is not None:
            label = "View: Orbit" if manager.viewing_through else "View: Scene Cam"
            if imgui.button(label):
                manager.toggle_viewport()
        else:
            imgui.text_disabled("No camera selected")

        imgui.separator()

        # --- Camera list ---
        for i, cam in enumerate(manager.cameras):
            is_active = i == manager.active_index
            flags = imgui.TreeNodeFlags_.default_open if is_active else 0

            # Highlight active camera
            if is_active:
                imgui.push_style_color(imgui.Col_.text, (1.0, 0.9, 0.3, 1.0))

            node_open = imgui.tree_node_ex(f"{cam.name}##cam_{i}", flags)

            if is_active:
                imgui.pop_style_color()

            if imgui.is_item_clicked():
                manager.select_camera(i)

            if node_open:
                # Position sliders
                changed_x, new_x = imgui.slider_float(
                    f"X##pos_{i}", float(cam.position[0]), -10.0, 10.0
                )
                changed_y, new_y = imgui.slider_float(
                    f"Y##pos_{i}", float(cam.position[1]), 0.0, 10.0
                )
                changed_z, new_z = imgui.slider_float(
                    f"Z##pos_{i}", float(cam.position[2]), -10.0, 10.0
                )
                if changed_x:
                    cam.position[0] = new_x
                if changed_y:
                    cam.position[1] = new_y
                if changed_z:
                    cam.position[2] = new_z

                # Rotation sliders (degrees for readability)
                import math as _math

                pitch_deg = float(_math.degrees(cam.rotation_euler[0]))
                yaw_deg = float(_math.degrees(cam.rotation_euler[1]))
                roll_deg = float(_math.degrees(cam.rotation_euler[2]))

                changed_p, new_p = imgui.slider_float(f"Pitch##rot_{i}", pitch_deg, -89.0, 89.0)
                changed_y, new_y = imgui.slider_float(f"Yaw##rot_{i}", yaw_deg, -180.0, 180.0)
                changed_r, new_r = imgui.slider_float(f"Roll##rot_{i}", roll_deg, -180.0, 180.0)
                if changed_p:
                    cam.rotation_euler[0] = _math.radians(new_p)
                if changed_y:
                    cam.rotation_euler[1] = _math.radians(new_y)
                if changed_r:
                    cam.rotation_euler[2] = _math.radians(new_r)

                # Camera format preset dropdown
                camera_formats = {
                    "Custom": (cam.fov_y, cam.aspect_ratio),
                    "Smartphone (9:16)": (45.0, 9.0 / 16.0),
                    "Smartphone (16:9)": (45.0, 16.0 / 9.0),
                    "Standard 4:3": (50.0, 4.0 / 3.0),
                    "Standard 3:4": (50.0, 3.0 / 4.0),
                    "HD Landscape (16:9)": (55.0, 16.0 / 9.0),
                    "HD Portrait (9:16)": (55.0, 9.0 / 16.0),
                    "Square (1:1)": (60.0, 1.0),
                    "Film (2.39:1)": (40.0, 2.39),
                    "Film (1:2.39)": (40.0, 1.0 / 2.39),
                    "Professional 3:2": (52.0, 3.0 / 2.0),
                    "Professional 2:3": (52.0, 2.0 / 3.0),
                }

                format_names = list(camera_formats.keys())
                current_format_idx = 0

                # Find best matching preset
                for idx, (_name, (fov, aspect)) in enumerate(camera_formats.items()):
                    if (
                        idx > 0
                        and abs(fov - cam.fov_y) < 0.1
                        and abs(aspect - cam.aspect_ratio) < 0.001
                    ):
                        current_format_idx = idx
                        break

                changed_format, new_format_idx = imgui.combo(
                    f"Format##fmt_{i}", current_format_idx, format_names
                )
                if changed_format and new_format_idx > 0:  # Skip "Custom"
                    fov, aspect = camera_formats[format_names[new_format_idx]]
                    cam.set_camera_format(fov, aspect)

                # FOV display
                changed_fov, new_fov = imgui.slider_float(f"FOV##fov_{i}", cam.fov_y, 10.0, 120.0)
                if changed_fov:
                    cam.fov_y = new_fov
                    cam._frustum_dirty = True

                # Visibility toggles
                _, cam.visible = imgui.checkbox(f"Visible##vis_{i}", cam.visible)
                imgui.same_line()
                _, cam.show_frustum = imgui.checkbox(f"Frustum##frust_{i}", cam.show_frustum)

                # Look-at button
                if imgui.button(f"Look at origin##look_{i}"):
                    import numpy as np

                    cam.look_at(np.array([0.0, 1.0, 0.0], dtype=np.float64))

                # Remove button
                imgui.same_line()
                if imgui.button(f"Remove##del_{i}"):
                    manager.remove_camera(i)
                    imgui.tree_pop()
                    break

                imgui.tree_pop()

        imgui.separator()

        # --- World axes toggle ---
        _, manager.show_world_axes = imgui.checkbox("Show world axes", manager.show_world_axes)

        imgui.separator()

        # --- Keyboard shortcuts help ---
        imgui.text_disabled("Tab: toggle panel")
        imgui.text_disabled("Middle-click: toggle panel")
        imgui.text_disabled("1-9: select scene camera")
        imgui.text_disabled("V: toggle orbit/scene view")
        imgui.text_disabled("WASD/Arrows: move (horizontal)")
        imgui.text_disabled("Space/Shift+Space: up/down")
        imgui.text_disabled("Shift: double speed")
        imgui.text_disabled("Scroll: zoom (FOV)")
        imgui.text_disabled("R-click drag: roll")
        imgui.text_disabled("Double R-click: level horizon")
        imgui.text_disabled("Ctrl: freeze orientation")

        imgui.end()
