"""Ground-plane grid mesh and world coordinate axes for spatial reference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def create_grid(
    size: float = 10.0,
    divisions: int = 20,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate a flat grid of lines on the XZ plane at Y=0.

    Args:
        size: Half-extent of the grid.
        divisions: Number of divisions along each axis.

    Returns:
        Tuple of (positions, colors) where:
        - positions: (N, 3) float32 line-segment endpoints
        - colors: (N, 3) float32 per-vertex colours (grey, with axes highlighted)
    """
    positions = []
    colors = []

    step = (2 * size) / divisions
    grey = [0.35, 0.35, 0.35]
    axis_x_color = [0.8, 0.2, 0.2]  # Red for X axis
    axis_z_color = [0.2, 0.2, 0.8]  # Blue for Z axis

    for i in range(divisions + 1):
        coord = -size + i * step

        # Lines parallel to Z axis
        color = axis_x_color if abs(coord) < step * 0.01 else grey
        positions.append([-size, 0.0, coord])
        positions.append([size, 0.0, coord])
        colors.append(color)
        colors.append(color)

        # Lines parallel to X axis
        color = axis_z_color if abs(coord) < step * 0.01 else grey
        positions.append([coord, 0.0, -size])
        positions.append([coord, 0.0, size])
        colors.append(color)
        colors.append(color)

    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32),
    )


def create_world_axes(
    length: float = 2.0,
    origin: tuple[float, float, float] = (0.0, 0.005, 0.0),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate coloured line segments for the world coordinate axes.

    Draws three axis lines from the origin:
    - +X: red
    - +Y: green
    - +Z: blue

    The origin is slightly above Y=0 to avoid z-fighting with the grid.

    Args:
        length: Length of each axis line.
        origin: World-space origin point for the axes.

    Returns:
        Tuple of (positions, colors) for GL_LINES rendering:
        - positions: (6, 3) float32 — two vertices per axis
        - colors: (6, 3) float32 — per-vertex RGB
    """
    ox, oy, oz = origin

    positions = np.array(
        [
            # +X axis
            [ox, oy, oz],
            [ox + length, oy, oz],
            # +Y axis
            [ox, oy, oz],
            [ox, oy + length, oz],
            # +Z axis
            [ox, oy, oz],
            [ox, oy, oz + length],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            # +X: red
            [1.0, 0.2, 0.2],
            [1.0, 0.2, 0.2],
            # +Y: green
            [0.2, 1.0, 0.2],
            [0.2, 1.0, 0.2],
            # +Z: blue
            [0.3, 0.3, 1.0],
            [0.3, 0.3, 1.0],
        ],
        dtype=np.float32,
    )

    return positions, colors
