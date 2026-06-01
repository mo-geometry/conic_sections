"""Tests for the procedural camera mesh generator."""

from __future__ import annotations

import numpy as np

from conic_sections.assets.camera_mesh import create_camera_mesh, create_frustum_lines
from conic_sections.assets.grid import create_world_axes


class TestCreateCameraMesh:
    """Camera mesh geometry tests."""

    def test_returns_four_arrays(self) -> None:
        """create_camera_mesh returns (positions, normals, colors, indices)."""
        result = create_camera_mesh()
        assert len(result) == 4

    def test_positions_shape(self) -> None:
        """Positions should be (N, 3) float32."""
        pos, _, _, _ = create_camera_mesh()
        assert pos.ndim == 2
        assert pos.shape[1] == 3
        assert pos.dtype == np.float32

    def test_normals_shape_matches_positions(self) -> None:
        """Normals should match positions shape."""
        pos, norms, _, _ = create_camera_mesh()
        assert norms.shape == pos.shape
        assert norms.dtype == np.float32

    def test_colors_shape_matches_positions(self) -> None:
        """Per-vertex colors should match positions shape."""
        pos, _, cols, _ = create_camera_mesh()
        assert cols.shape == pos.shape
        assert cols.dtype == np.float32

    def test_indices_dtype(self) -> None:
        """Indices should be uint32."""
        _, _, _, idx = create_camera_mesh()
        assert idx.dtype == np.uint32

    def test_indices_in_range(self) -> None:
        """All indices should reference valid vertices."""
        pos, _, _, idx = create_camera_mesh()
        assert np.all(idx < pos.shape[0])

    def test_indices_multiple_of_three(self) -> None:
        """Index count should be a multiple of 3 (triangles)."""
        _, _, _, idx = create_camera_mesh()
        assert idx.shape[0] % 3 == 0

    def test_normals_unit_length(self) -> None:
        """All normals should be approximately unit length."""
        _, norms, _, _ = create_camera_mesh()
        lengths = np.linalg.norm(norms, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.01)

    def test_mesh_centred_near_origin(self) -> None:
        """Mesh centre of mass should be near the origin."""
        pos, _, _, _ = create_camera_mesh()
        centroid = pos.mean(axis=0)
        assert np.linalg.norm(centroid) < 1.0

    def test_custom_body_dimensions(self) -> None:
        """Custom body dimensions should change the bounding box."""
        pos_small, _, _, _ = create_camera_mesh(body_width=0.3, body_height=0.2)
        pos_large, _, _, _ = create_camera_mesh(body_width=1.0, body_height=0.8)
        extent_small = float(pos_small[:, 0].max() - pos_small[:, 0].min())
        extent_large = float(pos_large[:, 0].max() - pos_large[:, 0].min())
        assert extent_large > extent_small

    def test_lens_segments_affect_vertex_count(self) -> None:
        """More lens segments should produce more vertices."""
        pos_8, _, _, _ = create_camera_mesh(lens_segments=8)
        pos_32, _, _, _ = create_camera_mesh(lens_segments=32)
        assert pos_32.shape[0] > pos_8.shape[0]

    def test_colors_in_valid_range(self) -> None:
        """Colors should be in [0, 1]."""
        _, _, cols, _ = create_camera_mesh()
        assert np.all(cols >= 0.0)
        assert np.all(cols <= 1.0)


class TestCreateFrustumLines:
    """Frustum wireframe geometry tests."""

    def test_returns_two_arrays(self) -> None:
        """create_frustum_lines returns (positions, colors)."""
        result = create_frustum_lines()
        assert len(result) == 2

    def test_positions_shape(self) -> None:
        """Positions should be (N, 3) float32 with even N (line pairs)."""
        pos, _ = create_frustum_lines()
        assert pos.ndim == 2
        assert pos.shape[1] == 3
        assert pos.shape[0] % 2 == 0
        assert pos.dtype == np.float32

    def test_colors_match_positions(self) -> None:
        """Colors array should match positions shape."""
        pos, cols = create_frustum_lines()
        assert cols.shape == pos.shape

    def test_fov_affects_extent(self) -> None:
        """Wider FOV should produce larger frustum extents."""
        pos_narrow, _ = create_frustum_lines(fov_y=30.0)
        pos_wide, _ = create_frustum_lines(fov_y=90.0)
        # Check the far plane extents (X range)
        narrow_extent = float(pos_narrow[:, 0].max() - pos_narrow[:, 0].min())
        wide_extent = float(pos_wide[:, 0].max() - pos_wide[:, 0].min())
        assert wide_extent > narrow_extent

    def test_custom_color(self) -> None:
        """Custom color should be applied to all vertices."""
        _, cols = create_frustum_lines(color=(1.0, 0.0, 0.0))
        np.testing.assert_allclose(cols[:, 0], 1.0)
        np.testing.assert_allclose(cols[:, 1], 0.0)
        np.testing.assert_allclose(cols[:, 2], 0.0)

    def test_frustum_extends_along_positive_z(self) -> None:
        """Frustum should extend along +Z (camera looks along +Z)."""
        pos, _ = create_frustum_lines(near=0.3, far=2.0)
        # All non-origin Z values should be positive
        non_origin = pos[np.any(pos != 0.0, axis=1)]
        assert np.all(non_origin[:, 2] > 0.0)


class TestCreateWorldAxes:
    """World coordinate axes geometry tests."""

    def test_returns_two_arrays(self) -> None:
        """create_world_axes returns (positions, colors)."""
        result = create_world_axes()
        assert len(result) == 2

    def test_positions_shape(self) -> None:
        """Should have 6 vertices (2 per axis)."""
        pos, _ = create_world_axes()
        assert pos.shape == (6, 3)
        assert pos.dtype == np.float32

    def test_colors_shape(self) -> None:
        """Colors should match positions."""
        pos, cols = create_world_axes()
        assert cols.shape == pos.shape

    def test_axes_extend_positive(self) -> None:
        """Each axis line should extend in the positive direction."""
        pos, _ = create_world_axes(length=2.0, origin=(0.0, 0.0, 0.0))
        # X axis endpoint
        assert pos[1, 0] == 2.0
        # Y axis endpoint
        assert pos[3, 1] == 2.0
        # Z axis endpoint
        assert pos[5, 2] == 2.0
