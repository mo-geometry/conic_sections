"""ModernGL rendering context and window management via GLFW."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl


def create_standalone_context(
    width: int = 1280,
    height: int = 720,
) -> moderngl.Context:
    """Create a headless (standalone) ModernGL context.

    Useful for off-screen rendering, testing, and CI environments
    where no display is available.

    Args:
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.

    Returns:
        A ModernGL context configured for off-screen rendering.
    """
    import moderngl

    ctx = moderngl.create_standalone_context()
    ctx.viewport = (0, 0, width, height)
    ctx.enable(moderngl.DEPTH_TEST)
    return ctx
