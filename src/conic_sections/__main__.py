"""ModernGL proof-of-concept — GLFW windowed renderer.

Renders a textured cube with Blinn-Phong shading, a ground-plane grid,
and mouse-driven orbit camera controls.

Run with:
    python -m conic_sections
"""

from __future__ import annotations

import sys

import glfw
import moderngl
import numpy as np

from conic_sections.assets.grid import create_grid
from conic_sections.assets.mesh import create_charuco_texture, create_cube
from conic_sections.core.orbit_camera import OrbitCamera
from conic_sections.rendering.shaders import (
    FRAGMENT_SHADER,
    GRID_FRAGMENT_SHADER,
    GRID_VERTEX_SHADER,
    VERTEX_SHADER,
)

# Window dimensions
WIDTH = 1280
HEIGHT = 720
TITLE = "Conic Sections — ModernGL PoC"


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

    # --- Compile shaders ---
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    grid_prog = ctx.program(vertex_shader=GRID_VERTEX_SHADER, fragment_shader=GRID_FRAGMENT_SHADER)

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

    # --- Camera setup ---
    camera = OrbitCamera(distance=5.0, azimuth=0.5, elevation=0.4)

    # Projection matrix
    aspect = fb_width / fb_height
    fov_y = float(np.radians(60.0))
    near, far = 0.1, 100.0
    proj = _perspective(fov_y, aspect, near, far)

    # Model matrix — translate cube up so its base sits on the grid
    model = np.eye(4, dtype=np.float32)
    model[1, 3] = 1.0  # ty = 1.0 (row-major: row 1, col 3)

    # --- GLFW callbacks ---
    def _mouse_button_cb(_win: object, button: int, action: int, mods: int) -> None:
        camera.on_mouse_button(button, action, mods)

    def _cursor_pos_cb(_win: object, x: float, y: float) -> None:
        camera.on_cursor_pos(x, y)

    def _scroll_cb(_win: object, x_offset: float, y_offset: float) -> None:
        camera.on_scroll(x_offset, y_offset)

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
    glfw.set_framebuffer_size_callback(window, _framebuffer_size_cb)

    # --- Light setup ---
    light_pos = np.array([5.0, 8.0, 5.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    object_color = np.array([0.6, 0.65, 0.75], dtype=np.float32)

    # --- Render loop ---
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Handle Escape key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        ctx.clear(0.12, 0.12, 0.14, 1.0)

        # Build column-major view matrix for OpenGL
        view = camera.view_matrix.astype(np.float32).T.copy()
        eye_pos = camera.eye.astype(np.float32)

        # Column-major model and projection
        model_cm = model.T.copy()
        proj_cm = proj.T.copy()

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

        glfw.swap_buffers(window)

    # --- Cleanup ---
    vao.release()
    vbo_pos.release()
    vbo_norm.release()
    vbo_tex.release()
    ibo.release()
    texture.release()
    grid_vao.release()
    grid_vbo_pos.release()
    grid_vbo_col.release()
    prog.release()
    grid_prog.release()
    ctx.release()
    glfw.terminate()


if __name__ == "__main__":
    main()
