"""ModernGL renderer with scene camera placement and Dear ImGui controls.

Renders a textured cube with Blinn-Phong shading, a ground-plane grid,
and one or more poseable camera models in the scene.  Dear ImGui provides
a control panel for adding, positioning, and selecting cameras.  The
viewport can toggle between the orbit camera and any scene camera.

Controls:
    Tab / Middle-click  — toggle ImGui panel (mouse goes to panel or orbit)
    1-9                 — select scene camera (when panel hidden)
    V                   — toggle orbit / scene camera viewport
    WASD / Arrows       — move scene camera on horizontal plane
    Space / Shift+Space — ascend / descend
    Shift               — double movement speed
    Mouse               — yaw / pitch (look around) in scene camera mode
    Right-click + drag  — roll (tilt) in scene camera mode
    Scroll              — zoom (change FOV) in scene camera mode
    Double right-click  — reset horizon (level roll)
    Ctrl (hold)         — freeze camera orientation (move mouse to ImGui)
    Escape              — quit

Run with:
    python -m conic_sections
"""

from __future__ import annotations

import sys

import glfw
import moderngl
import numpy as np

from conic_sections.assets.camera_mesh import create_camera_mesh, create_frustum_lines
from conic_sections.assets.grid import create_grid, create_world_axes
from conic_sections.assets.mesh import create_charuco_texture, create_cube
from conic_sections.core.orbit_camera import OrbitCamera
from conic_sections.core.scene_camera import SceneCameraManager
from conic_sections.rendering.imgui_layer import ImGuiLayer
from conic_sections.rendering.shaders import (
    FRAGMENT_SHADER,
    GRID_FRAGMENT_SHADER,
    GRID_VERTEX_SHADER,
    VERTEX_SHADER,
)

# Window dimensions
WIDTH = 1280
HEIGHT = 720
TITLE = "Conic Sections — v0.3.0"


# ---------------------------------------------------------------------------
# Shader for camera mesh (per-vertex colour, no texture)
# ---------------------------------------------------------------------------

CAMERA_VERTEX_SHADER = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 frag_pos;
out vec3 frag_normal;
out vec3 frag_color_v;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_color_v = in_color;
    gl_Position = projection * view * world_pos;
}
"""

CAMERA_FRAGMENT_SHADER = """
#version 330 core

uniform vec3 light_pos;
uniform vec3 light_color;
uniform vec3 view_pos;

in vec3 frag_pos;
in vec3 frag_normal;
in vec3 frag_color_v;

out vec4 frag_color;

void main() {
    // Ambient
    float ambient_strength = 0.15;
    vec3 ambient = ambient_strength * light_color;

    // Diffuse
    vec3 norm = normalize(frag_normal);
    vec3 light_dir = normalize(light_pos - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    // Specular (Blinn-Phong)
    float specular_strength = 0.3;
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 halfway = normalize(light_dir + view_dir);
    float spec = pow(max(dot(norm, halfway), 0.0), 32.0);
    vec3 specular = specular_strength * spec * light_color;

    vec3 result = (ambient + diffuse + specular) * frag_color_v;
    frag_color = vec4(result, 1.0);
}
"""


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Build a column-major 4x4 perspective projection matrix."""
    f = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[3, 2] = -1.0
    proj[2, 3] = (2.0 * far * near) / (near - far)
    return proj


def main() -> None:
    """Launch the windowed renderer."""
    # --- GLFW initialisation ---
    if not glfw.init():
        sys.exit("Failed to initialise GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
    if not window:
        glfw.terminate()
        sys.exit("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Vsync

    # --- ModernGL context ---
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)

    # Query actual framebuffer size (Retina displays are 2x window size)
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fb_width, fb_height)

    # Print OpenGL info for diagnostics
    print(f"OpenGL: {ctx.info['GL_VERSION']}")
    print(f"Renderer: {ctx.info['GL_RENDERER']}")
    print(f"Framebuffer: {fb_width}x{fb_height}")

    # --- Dear ImGui ---
    imgui_layer = ImGuiLayer(window)

    # --- Compile shaders ---
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    grid_prog = ctx.program(vertex_shader=GRID_VERTEX_SHADER, fragment_shader=GRID_FRAGMENT_SHADER)
    cam_prog = ctx.program(
        vertex_shader=CAMERA_VERTEX_SHADER, fragment_shader=CAMERA_FRAGMENT_SHADER
    )

    # --- Create cube mesh ---
    positions, normals, texcoords, indices = create_cube(size=1.0)

    vbo_pos = ctx.buffer(positions.tobytes())
    vbo_norm = ctx.buffer(normals.tobytes())
    vbo_tex = ctx.buffer(texcoords.tobytes())
    ibo = ctx.buffer(indices.tobytes())

    vao = ctx.vertex_array(
        prog,
        [
            (vbo_pos, "3f", "in_position"),
            (vbo_norm, "3f", "in_normal"),
            (vbo_tex, "2f", "in_texcoord"),
        ],
        index_buffer=ibo,
        index_element_size=4,
    )

    # --- Create ChArUco texture ---
    tex_data = create_charuco_texture(squares_x=8, squares_y=8, resolution=512)
    texture = ctx.texture((512, 512), 3, tex_data.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.build_mipmaps()

    # --- Create ground grid ---
    grid_positions, grid_colors = create_grid(size=10.0, divisions=20)
    grid_vbo_pos = ctx.buffer(grid_positions.tobytes())
    grid_vbo_col = ctx.buffer(grid_colors.tobytes())

    grid_vao = ctx.vertex_array(
        grid_prog,
        [
            (grid_vbo_pos, "3f", "in_position"),
            (grid_vbo_col, "3f", "in_color"),
        ],
    )

    # --- Create world coordinate axes ---
    axes_positions, axes_colors = create_world_axes(length=2.0)
    axes_vbo_pos = ctx.buffer(axes_positions.tobytes())
    axes_vbo_col = ctx.buffer(axes_colors.tobytes())

    axes_vao = ctx.vertex_array(
        grid_prog,
        [
            (axes_vbo_pos, "3f", "in_position"),
            (axes_vbo_col, "3f", "in_color"),
        ],
    )

    # --- Create camera mesh (shared geometry for all scene cameras) ---
    cam_positions, cam_normals, cam_colors, cam_indices = create_camera_mesh()

    cam_vbo_pos = ctx.buffer(cam_positions.tobytes())
    cam_vbo_norm = ctx.buffer(cam_normals.tobytes())
    cam_vbo_col = ctx.buffer(cam_colors.tobytes())
    cam_ibo = ctx.buffer(cam_indices.tobytes())

    cam_vao = ctx.vertex_array(
        cam_prog,
        [
            (cam_vbo_pos, "3f", "in_position"),
            (cam_vbo_norm, "3f", "in_normal"),
            (cam_vbo_col, "3f", "in_color"),
        ],
        index_buffer=cam_ibo,
        index_element_size=4,
    )

    # --- Create frustum line geometry (shared, transformed per camera) ---
    frust_positions, frust_colors = create_frustum_lines()
    frust_vbo_pos = ctx.buffer(frust_positions.tobytes())
    frust_vbo_col = ctx.buffer(frust_colors.tobytes())

    # Line shader with model matrix support for frustum wireframe
    line_model_vs = """
    #version 330 core
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    in vec3 in_position;
    in vec3 in_color;
    out vec3 frag_color_v;
    void main() {
        frag_color_v = in_color;
        gl_Position = projection * view * model * vec4(in_position, 1.0);
    }
    """
    line_model_fs = """
    #version 330 core
    in vec3 frag_color_v;
    out vec4 frag_color;
    void main() {
        frag_color = vec4(frag_color_v, 1.0);
    }
    """
    line_model_prog = ctx.program(vertex_shader=line_model_vs, fragment_shader=line_model_fs)

    frust_vao = ctx.vertex_array(
        line_model_prog,
        [
            (frust_vbo_pos, "3f", "in_position"),
            (frust_vbo_col, "3f", "in_color"),
        ],
    )

    # --- Camera setup ---
    orbit_camera = OrbitCamera(distance=5.0, azimuth=0.5, elevation=0.4)
    camera_manager = SceneCameraManager()

    # Projection matrix
    aspect = fb_width / fb_height
    fov_y = float(np.radians(60.0))
    near, far = 0.1, 100.0
    proj = _perspective(fov_y, aspect, near, far)

    # Model matrix — translate cube up so its base sits on the grid
    model = np.eye(4, dtype=np.float32)
    model[1, 3] = 1.0  # ty = 1.0 (row-major: row 1, col 3)

    # --- GLFW callbacks ---
    # All callbacks forward to ImGui first, then conditionally to orbit camera.

    orientation_frozen = False  # Ctrl toggles this on/off

    def _mouse_button_cb(win: object, button: int, action: int, mods: int) -> None:
        # Forward to ImGui so it can track hover state
        imgui_layer.on_mouse_button(win, button, action, mods)

        # Middle-click toggles the panel
        if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            imgui_layer.toggle_panel()
            return

        # Only pass to camera controls if ImGui doesn't want the mouse
        if not imgui_layer.want_capture_mouse:
            active = camera_manager.active_camera
            if camera_manager.viewing_through and active is not None:
                active.on_mouse_button(button, action, mods)
            else:
                orbit_camera.on_mouse_button(button, action, mods)

    def _cursor_pos_cb(win: object, x: float, y: float) -> None:
        # Forward to ImGui so it can track hover state
        imgui_layer.on_cursor_pos(win, x, y)

        active = camera_manager.active_camera
        if orientation_frozen or imgui_layer.want_capture_mouse:
            # Sync both cameras so neither jumps on resume
            orbit_camera.sync_cursor(x, y)
            if active is not None:
                active.sync_cursor(x, y)
        elif camera_manager.viewing_through and active is not None:
            active.on_cursor_pos(x, y)
            # Keep orbit camera in sync so there's no jump on switch-back
            orbit_camera.sync_cursor(x, y)
        else:
            orbit_camera.on_cursor_pos(x, y)
            # Keep scene camera in sync
            if active is not None:
                active.sync_cursor(x, y)

    def _scroll_cb(win: object, x_offset: float, y_offset: float) -> None:
        # Forward to ImGui
        imgui_layer.on_scroll(win, x_offset, y_offset)

        if not imgui_layer.want_capture_mouse:
            active = camera_manager.active_camera
            if camera_manager.viewing_through and active is not None:
                active.on_scroll(x_offset, y_offset)
            else:
                orbit_camera.on_scroll(x_offset, y_offset)

    def _update_cursor_mode() -> None:
        """Set GLFW cursor mode based on whether we're viewing through a camera.

        The cursor is captured (disabled) only when viewing through a scene
        camera AND the ImGui panel is not visible AND orientation is not
        frozen — otherwise the user couldn't interact with ImGui widgets.
        """
        viewing = (
            camera_manager.viewing_through
            and camera_manager.active_camera is not None
            and not imgui_layer.panel_visible
            and not orientation_frozen
        )
        if viewing:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def _key_cb(
        win: object,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        # Forward to ImGui
        imgui_layer.on_key(win, key, scancode, action, mods)

        if action != glfw.PRESS:
            return

        # Ctrl toggles orientation freeze (mouse stops rotating camera)
        if key in (glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL):
            nonlocal orientation_frozen
            orientation_frozen = not orientation_frozen
            _update_cursor_mode()
            return

        # Tab toggles the ImGui panel
        if key == glfw.KEY_TAB:
            imgui_layer.toggle_panel()
            return

        # Escape: if viewing through camera, exit to orbit; otherwise quit
        if key == glfw.KEY_ESCAPE:
            if camera_manager.viewing_through:
                camera_manager.viewing_through = False
                _update_cursor_mode()
            else:
                glfw.set_window_should_close(window, True)
            return

        # Below here: only when panel is hidden (orbit mode)
        if imgui_layer.want_capture_keyboard:
            return

        # V toggles orbit/scene camera viewport
        if key == glfw.KEY_V:
            camera_manager.toggle_viewport()
            _update_cursor_mode()

        # 1-9 selects scene cameras
        if glfw.KEY_1 <= key <= glfw.KEY_9:
            idx = key - glfw.KEY_1
            if idx < len(camera_manager.cameras):
                camera_manager.select_camera(idx)

    def _char_cb(win: object, char: int) -> None:
        imgui_layer.on_char(win, char)

    def _framebuffer_size_cb(_win: object, width: int, height: int) -> None:
        nonlocal proj, aspect
        if height == 0:
            return
        ctx.viewport = (0, 0, width, height)
        aspect = width / height
        proj = _perspective(fov_y, aspect, near, far)

    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window, _cursor_pos_cb)
    glfw.set_scroll_callback(window, _scroll_cb)
    glfw.set_key_callback(window, _key_cb)
    glfw.set_char_callback(window, _char_cb)
    glfw.set_framebuffer_size_callback(window, _framebuffer_size_cb)

    # --- Light setup ---
    light_pos = np.array([5.0, 8.0, 5.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    object_color = np.array([0.6, 0.65, 0.75], dtype=np.float32)

    # --- Render loop ---
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # --- Update cursor mode (handles ImGui button toggles too) ---
        _update_cursor_mode()

        # --- FPS movement (poll keys every frame) ---
        active = camera_manager.active_camera
        if camera_manager.viewing_through and active is not None:
            shift_held = (
                glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
            fwd = 0.0
            strafe = 0.0
            vert = 0.0
            if (
                glfw.get_key(window, glfw.KEY_W) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
            ):
                fwd += 1.0
            if (
                glfw.get_key(window, glfw.KEY_S) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
            ):
                fwd -= 1.0
            if (
                glfw.get_key(window, glfw.KEY_D) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
            ):
                strafe -= 1.0
            if (
                glfw.get_key(window, glfw.KEY_A) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
            ):
                strafe += 1.0
            # Space = ascend, Shift+Space = descend
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                vert += -1.0 if shift_held else 1.0
            if fwd != 0.0 or strafe != 0.0 or vert != 0.0:
                active.move(
                    forward=fwd,
                    strafe=strafe,
                    vertical=vert,
                    fast=shift_held,
                )

        # --- ImGui frame ---
        imgui_layer.new_frame()
        imgui_layer.render_camera_panel(camera_manager)

        ctx.clear(0.12, 0.12, 0.14, 1.0)

        # --- Determine active view/projection ---
        orbit_view = orbit_camera.view_matrix.astype(np.float64)
        orbit_proj_f64 = proj.astype(np.float64)

        view_f64, proj_f64, eye_f64 = camera_manager.get_view_projection(
            orbit_view, orbit_proj_f64
        )

        # Convert to float32, column-major for OpenGL
        view = view_f64.astype(np.float32).T.copy()
        proj_cm = proj_f64.astype(np.float32).T.copy()
        eye_pos = eye_f64.astype(np.float32)

        # Column-major model
        model_cm = model.T.copy()

        # --- Draw cube ---
        prog["model"].write(model_cm.tobytes())
        prog["view"].write(view.tobytes())
        prog["projection"].write(proj_cm.tobytes())
        prog["light_pos"].write(light_pos.tobytes())
        prog["light_color"].write(light_color.tobytes())
        prog["view_pos"].write(eye_pos.tobytes())

        # Bind texture before any draw calls to avoid macOS driver warning
        texture.use(location=0)
        prog["tex"].value = 0

        # Draw front face with ChArUco texture
        prog["use_texture"].value = True
        vao.render(moderngl.TRIANGLES, vertices=6, first=0)

        # Draw remaining faces with solid colour
        prog["use_texture"].value = False
        prog["object_color"].write(object_color.tobytes())
        vao.render(moderngl.TRIANGLES, vertices=30, first=6)

        # --- Draw grid ---
        grid_prog["view"].write(view.tobytes())
        grid_prog["projection"].write(proj_cm.tobytes())
        grid_vao.render(moderngl.LINES)

        # --- Draw world coordinate axes ---
        # Note: glLineWidth > 1.0 is not supported in macOS core profile,
        # so we render at the default width of 1.0.
        if camera_manager.show_world_axes:
            axes_vao.render(moderngl.LINES)

        # --- Draw scene cameras ---
        for i, scene_cam in enumerate(camera_manager.cameras):
            if not scene_cam.visible:
                continue

            # Don't draw the camera we're looking through
            if camera_manager.viewing_through and i == camera_manager.active_index:
                continue

            # Camera mesh model matrix
            cam_model = scene_cam.model_matrix.astype(np.float32).T.copy()

            cam_prog["model"].write(cam_model.tobytes())
            cam_prog["view"].write(view.tobytes())
            cam_prog["projection"].write(proj_cm.tobytes())
            cam_prog["light_pos"].write(light_pos.tobytes())
            cam_prog["light_color"].write(light_color.tobytes())
            cam_prog["view_pos"].write(eye_pos.tobytes())

            cam_vao.render(moderngl.TRIANGLES)

            # Draw frustum wireframe (regenerated per camera to track FOV)
            if scene_cam.show_frustum:
                cam_aspect = scene_cam.camera.width / scene_cam.camera.height
                fp, fc = create_frustum_lines(
                    fov_y=scene_cam.fov_y,
                    aspect=cam_aspect,
                )
                frust_vbo_pos.orphan(fp.nbytes)
                frust_vbo_pos.write(fp.tobytes())
                frust_vbo_col.orphan(fc.nbytes)
                frust_vbo_col.write(fc.tobytes())

                line_model_prog["model"].write(cam_model.tobytes())
                line_model_prog["view"].write(view.tobytes())
                line_model_prog["projection"].write(proj_cm.tobytes())
                frust_vao.render(moderngl.LINES)

        # --- ImGui render ---
        imgui_layer.render_draw_data()

        glfw.swap_buffers(window)

    # --- Cleanup ---
    imgui_layer.shutdown()

    vao.release()
    vbo_pos.release()
    vbo_norm.release()
    vbo_tex.release()
    ibo.release()
    texture.release()
    grid_vao.release()
    grid_vbo_pos.release()
    grid_vbo_col.release()
    axes_vao.release()
    axes_vbo_pos.release()
    axes_vbo_col.release()
    cam_vao.release()
    cam_vbo_pos.release()
    cam_vbo_norm.release()
    cam_vbo_col.release()
    cam_ibo.release()
    frust_vao.release()
    frust_vbo_pos.release()
    frust_vbo_col.release()
    prog.release()
    grid_prog.release()
    cam_prog.release()
    line_model_prog.release()
    ctx.release()
    glfw.terminate()


if __name__ == "__main__":
    main()
