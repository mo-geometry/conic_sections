"""Procedural mesh generation for primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def create_cube(
    size: float = 1.0,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
]:
    """Generate a cube mesh with per-face normals and texture coordinates.

    Each face has 4 unique vertices (24 total) so normals are sharp.
    The front face (+Z) is designated for the ChArUco texture.

    Args:
        size: Half-extent of the cube.

    Returns:
        Tuple of (positions, normals, texcoords, indices) where:
        - positions: (24, 3) float32 vertex positions
        - normals: (24, 3) float32 per-vertex normals
        - texcoords: (24, 2) float32 UV coordinates
        - indices: (36,) uint32 triangle indices
    """
    s = size

    # fmt: off
    # Each face: 4 vertices with position, normal, and UV
    # Face order: front(+Z), back(-Z), right(+X), left(-X), top(+Y), bottom(-Y)
    positions = np.array([
        # Front face (+Z)
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        # Back face (-Z)
        [ s, -s, -s], [-s, -s, -s], [-s,  s, -s], [ s,  s, -s],
        # Right face (+X)
        [ s, -s,  s], [ s, -s, -s], [ s,  s, -s], [ s,  s,  s],
        # Left face (-X)
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
        # Top face (+Y)
        [-s,  s,  s], [ s,  s,  s], [ s,  s, -s], [-s,  s, -s],
        # Bottom face (-Y)
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
    ], dtype=np.float32)

    normals = np.array([
        # Front
        [ 0,  0,  1], [ 0,  0,  1], [ 0,  0,  1], [ 0,  0,  1],
        # Back
        [ 0,  0, -1], [ 0,  0, -1], [ 0,  0, -1], [ 0,  0, -1],
        # Right
        [ 1,  0,  0], [ 1,  0,  0], [ 1,  0,  0], [ 1,  0,  0],
        # Left
        [-1,  0,  0], [-1,  0,  0], [-1,  0,  0], [-1,  0,  0],
        # Top
        [ 0,  1,  0], [ 0,  1,  0], [ 0,  1,  0], [ 0,  1,  0],
        # Bottom
        [ 0, -1,  0], [ 0, -1,  0], [ 0, -1,  0], [ 0, -1,  0],
    ], dtype=np.float32)

    texcoords = np.array([
        # Each face gets full 0-1 UV range
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float32)

    # Two triangles per face, CCW winding
    indices = np.array([
        0,  1,  2,  0,  2,  3,   # Front
        4,  5,  6,  4,  6,  7,   # Back
        8,  9,  10, 8,  10, 11,  # Right
        12, 13, 14, 12, 14, 15,  # Left
        16, 17, 18, 16, 18, 19,  # Top
        20, 21, 22, 20, 22, 23,  # Bottom
    ], dtype=np.uint32)
    # fmt: on

    return positions, normals, texcoords, indices


def create_charuco_texture(
    squares_x: int = 8,
    squares_y: int = 8,
    resolution: int = 512,
) -> npt.NDArray[np.uint8]:
    """Generate a ChArUco-style checkerboard texture.

    Creates a black-and-white checkerboard pattern. The ArUco marker
    placement can be added later; this provides the base calibration
    pattern.

    Args:
        squares_x: Number of squares along the x-axis.
        squares_y: Number of squares along the y-axis.
        resolution: Texture resolution in pixels (square).

    Returns:
        An (resolution, resolution, 3) uint8 RGB texture array.
    """
    texture = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    sq_w = resolution // squares_x
    sq_h = resolution // squares_y

    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y0 = row * sq_h
                y1 = y0 + sq_h
                x0 = col * sq_w
                x1 = x0 + sq_w
                texture[y0:y1, x0:x1] = 255

    return texture
