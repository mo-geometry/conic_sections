"""Procedural camera mesh for scene placement.

Generates a recognisable 3D camera model composed of:
- A rectangular body (the "housing")
- A cylindrical lens barrel protruding from the front face
- A small viewfinder bump on top-rear

The mesh uses per-vertex colour so the front (lens) face is visually
distinct from the body, making camera orientation intuitive.  The camera
looks along its local +Z axis, with +Y up and +X right.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _box(
    sx: float,
    sy: float,
    sz: float,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    color: tuple[float, float, float] = (0.35, 0.35, 0.38),
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[int]]]:
    """Axis-aligned box with per-face normals.

    Args:
        sx, sy, sz: Half-extents along each axis.
        cx, cy, cz: Centre offset.
        color: RGB colour for all vertices.

    Returns:
        (positions, normals, colors, quads) — quads are 4-index face lists.
    """
    # Eight corners
    c = [
        [cx - sx, cy - sy, cz + sz],  # 0  front-bottom-left
        [cx + sx, cy - sy, cz + sz],  # 1  front-bottom-right
        [cx + sx, cy + sy, cz + sz],  # 2  front-top-right
        [cx - sx, cy + sy, cz + sz],  # 3  front-top-left
        [cx - sx, cy - sy, cz - sz],  # 4  back-bottom-left
        [cx + sx, cy - sy, cz - sz],  # 5  back-bottom-right
        [cx + sx, cy + sy, cz - sz],  # 6  back-top-right
        [cx - sx, cy + sy, cz - sz],  # 7  back-top-left
    ]

    # Six faces: (corner indices CCW from outside, outward normal)
    faces = [
        ([0, 1, 2, 3], [0.0, 0.0, 1.0]),  # +Z  front
        ([5, 4, 7, 6], [0.0, 0.0, -1.0]),  # -Z  back
        ([1, 5, 6, 2], [1.0, 0.0, 0.0]),  # +X  right
        ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),  # -X  left
        ([3, 2, 6, 7], [0.0, 1.0, 0.0]),  # +Y  top
        ([4, 5, 1, 0], [0.0, -1.0, 0.0]),  # -Y  bottom
    ]

    positions: list[list[float]] = []
    normals: list[list[float]] = []
    colors: list[list[float]] = []
    quads: list[list[int]] = []

    for indices, normal in faces:
        base = len(positions)
        for i in indices:
            positions.append(c[i])
            normals.append(normal)
            colors.append(list(color))
        quads.append([base, base + 1, base + 2, base + 3])

    return positions, normals, colors, quads


def _cylinder(
    radius: float,
    length: float,
    segments: int = 16,
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    color: tuple[float, float, float] = (0.25, 0.25, 0.28),
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[int]]]:
    """Cylinder along the Z axis with flat-shaded caps.

    The cylinder extends from cz to cz + length (front cap at cz + length,
    back cap at cz).

    Args:
        radius: Radius of the cylinder.
        length: Length along Z.
        segments: Number of circumferential divisions.
        cx, cy, cz: Centre of the back cap.
        color: RGB colour for all vertices.

    Returns:
        (positions, normals, colors, quads).
    """
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    colors: list[list[float]] = []
    quads: list[list[int]] = []

    z_back = cz
    z_front = cz + length

    # Pre-compute ring points
    angles = [2.0 * math.pi * i / segments for i in range(segments)]
    ring_x = [cx + radius * math.cos(a) for a in angles]
    ring_y = [cy + radius * math.sin(a) for a in angles]

    # --- Side faces ---
    for i in range(segments):
        j = (i + 1) % segments
        base = len(positions)
        # Quad: back-i, back-j, front-j, front-i
        pts = [
            [ring_x[i], ring_y[i], z_back],
            [ring_x[j], ring_y[j], z_back],
            [ring_x[j], ring_y[j], z_front],
            [ring_x[i], ring_y[i], z_front],
        ]
        for p in pts:
            positions.append(p)
            # Outward radial normal (approximate flat shading)
            mid_a = (angles[i] + angles[j]) / 2.0
            normals.append([math.cos(mid_a), math.sin(mid_a), 0.0])
            colors.append(list(color))
        quads.append([base, base + 1, base + 2, base + 3])

    # --- Front cap (+Z) ---
    front_center_idx = len(positions)
    positions.append([cx, cy, z_front])
    normals.append([0.0, 0.0, 1.0])
    colors.append(list(color))
    for i in range(segments):
        j = (i + 1) % segments
        base = len(positions)
        positions.append([ring_x[i], ring_y[i], z_front])
        normals.append([0.0, 0.0, 1.0])
        colors.append(list(color))
        positions.append([ring_x[j], ring_y[j], z_front])
        normals.append([0.0, 0.0, 1.0])
        colors.append(list(color))
        # Triangle fan: center, i, j  (stored as degenerate quad)
        quads.append([front_center_idx, base, base + 1, base + 1])

    # --- Back cap (-Z) ---
    back_center_idx = len(positions)
    positions.append([cx, cy, z_back])
    normals.append([0.0, 0.0, -1.0])
    colors.append(list(color))
    for i in range(segments):
        j = (i + 1) % segments
        base = len(positions)
        positions.append([ring_x[j], ring_y[j], z_back])
        normals.append([0.0, 0.0, -1.0])
        colors.append(list(color))
        positions.append([ring_x[i], ring_y[i], z_back])
        normals.append([0.0, 0.0, -1.0])
        colors.append(list(color))
        quads.append([back_center_idx, base, base + 1, base + 1])

    return positions, normals, colors, quads


def _quads_to_triangles(quads: list[list[int]]) -> list[int]:
    """Convert quad face lists to triangle index lists.

    Degenerate quads (where indices 2 == 3) produce a single triangle.
    """
    triangles: list[int] = []
    for q in quads:
        triangles.extend([q[0], q[1], q[2]])
        if q[2] != q[3]:
            triangles.extend([q[0], q[2], q[3]])
    return triangles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_camera_mesh(
    body_width: float = 0.6,
    body_height: float = 0.4,
    body_depth: float = 0.35,
    lens_radius: float = 0.15,
    lens_length: float = 0.25,
    lens_segments: int = 16,
    body_color: tuple[float, float, float] = (0.35, 0.35, 0.38),
    lens_color: tuple[float, float, float] = (0.20, 0.20, 0.22),
    accent_color: tuple[float, float, float] = (0.85, 0.25, 0.15),
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uint32],
]:
    """Generate a procedural camera mesh.

    The camera looks along its local +Z axis.  The lens barrel protrudes
    from the front face (+Z), aligned with the camera's look direction.

    Coordinate layout (local space, centred at origin):
    - Body centred at origin
    - Lens barrel extends from front face (+Z) outward
    - Viewfinder bump sits on top-rear of body

    Args:
        body_width: Full width of the camera body (X axis).
        body_height: Full height of the camera body (Y axis).
        body_depth: Full depth of the camera body (Z axis).
        lens_radius: Radius of the lens barrel cylinder.
        lens_length: How far the lens protrudes from the front face.
        lens_segments: Circumferential resolution of the lens cylinder.
        body_color: RGB colour for the camera housing.
        lens_color: RGB colour for the lens barrel.
        accent_color: RGB colour for the viewfinder and front face ring.

    Returns:
        Tuple of (positions, normals, colors, indices) where:
        - positions: (N, 3) float32 vertex positions
        - normals: (N, 3) float32 per-vertex normals
        - colors: (N, 3) float32 per-vertex RGB colours
        - indices: (M,) uint32 triangle indices
    """
    all_pos: list[list[float]] = []
    all_norm: list[list[float]] = []
    all_col: list[list[float]] = []
    all_quads: list[list[int]] = []

    def _merge(
        parts: tuple[list[list[float]], list[list[float]], list[list[float]], list[list[int]]],
    ) -> None:
        pos, nrm, col, quads = parts
        offset = len(all_pos)
        all_pos.extend(pos)
        all_norm.extend(nrm)
        all_col.extend(col)
        for q in quads:
            all_quads.append([i + offset for i in q])

    hw = body_width / 2.0
    hh = body_height / 2.0
    hd = body_depth / 2.0

    # 1. Main body
    _merge(_box(hw, hh, hd, color=body_color))

    # 2. Lens barrel — protrudes from the front face (+Z)
    _merge(
        _cylinder(
            radius=lens_radius,
            length=lens_length,
            segments=lens_segments,
            cx=0.0,
            cy=0.0,
            cz=hd,
            color=lens_color,
        )
    )

    # 3. Viewfinder bump — small box on top-rear
    vf_w = body_width * 0.25
    vf_h = body_height * 0.22
    vf_d = body_depth * 0.3
    _merge(
        _box(
            vf_w / 2.0,
            vf_h / 2.0,
            vf_d / 2.0,
            cx=0.0,
            cy=hh + vf_h / 2.0,
            cz=-hd + vf_d / 2.0 + 0.02,
            color=accent_color,
        )
    )

    # 4. Front face accent ring — thin frame around the lens mount
    ring_w = body_width * 0.85
    ring_h = body_height * 0.85
    ring_d = 0.02
    _merge(
        _box(
            ring_w / 2.0,
            ring_h / 2.0,
            ring_d / 2.0,
            cx=0.0,
            cy=0.0,
            cz=hd + ring_d / 2.0,
            color=accent_color,
        )
    )

    # Convert to numpy arrays
    positions = np.array(all_pos, dtype=np.float32)
    normals_arr = np.array(all_norm, dtype=np.float32)
    colors = np.array(all_col, dtype=np.float32)
    indices = np.array(_quads_to_triangles(all_quads), dtype=np.uint32)

    return positions, normals_arr, colors, indices


def create_frustum_lines(
    fov_y: float = 60.0,
    aspect: float = 16.0 / 9.0,
    near: float = 0.3,
    far: float = 2.0,
    color: tuple[float, float, float] = (1.0, 1.0, 0.3),
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate line-segment geometry for a camera view frustum.

    The frustum extends along the local +Z axis, matching the scene
    camera convention (the camera looks along +Z).  The frustum origin
    is at the camera centre; the model matrix of the SceneCamera
    transforms it into world space.

    Args:
        fov_y: Vertical field of view in degrees.
        aspect: Width / height aspect ratio.
        near: Near plane distance from camera.
        far: Far plane distance from camera.
        color: RGB colour for all frustum lines.

    Returns:
        Tuple of (positions, colors) for GL_LINES rendering:
        - positions: (N, 3) float32
        - colors: (N, 3) float32
    """
    half_v_near = near * math.tan(math.radians(fov_y / 2.0))
    half_h_near = half_v_near * aspect
    half_v_far = far * math.tan(math.radians(fov_y / 2.0))
    half_h_far = half_v_far * aspect

    # Near plane corners (at z = +near, camera looks along +Z)
    n_tl = [-half_h_near, half_v_near, near]
    n_tr = [half_h_near, half_v_near, near]
    n_br = [half_h_near, -half_v_near, near]
    n_bl = [-half_h_near, -half_v_near, near]

    # Far plane corners
    f_tl = [-half_h_far, half_v_far, far]
    f_tr = [half_h_far, half_v_far, far]
    f_br = [half_h_far, -half_v_far, far]
    f_bl = [-half_h_far, -half_v_far, far]

    # Line segments: 4 edges of near rect, 4 edges of far rect, 4 connecting
    lines = [
        # Near rectangle
        n_tl,
        n_tr,
        n_tr,
        n_br,
        n_br,
        n_bl,
        n_bl,
        n_tl,
        # Far rectangle
        f_tl,
        f_tr,
        f_tr,
        f_br,
        f_br,
        f_bl,
        f_bl,
        f_tl,
        # Connecting edges (origin to near corners)
        [0, 0, 0],
        n_tl,
        [0, 0, 0],
        n_tr,
        [0, 0, 0],
        n_br,
        [0, 0, 0],
        n_bl,
        # Near to far
        n_tl,
        f_tl,
        n_tr,
        f_tr,
        n_br,
        f_br,
        n_bl,
        f_bl,
    ]

    positions = np.array(lines, dtype=np.float32)
    colors = np.full_like(positions, color, dtype=np.float32)

    return positions, colors
