"""Tests for procedural mesh generation."""

from __future__ import annotations

import numpy as np

from conic_sections.assets.mesh import create_charuco_texture, create_cube


class TestCreateCube:
    """Cube mesh generation tests."""

    def test_vertex_count(self) -> None:
        positions, normals, texcoords, indices = create_cube()
        # 6 faces x 4 vertices = 24
        assert positions.shape == (24, 3)
        assert normals.shape == (24, 3)
        assert texcoords.shape == (24, 2)

    def test_index_count(self) -> None:
        _, _, _, indices = create_cube()
        # 6 faces x 2 triangles x 3 indices = 36
        assert indices.shape == (36,)

    def test_normals_are_unit_length(self) -> None:
        _, normals, _, _ = create_cube()
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    def test_indices_in_range(self) -> None:
        positions, _, _, indices = create_cube()
        assert indices.max() < len(positions)
        assert indices.min() >= 0

    def test_size_parameter(self) -> None:
        positions, _, _, _ = create_cube(size=2.0)
        assert positions.max() <= 2.0
        assert positions.min() >= -2.0

    def test_front_face_normal_is_positive_z(self) -> None:
        _, normals, _, _ = create_cube()
        # First 4 vertices are the front face
        for i in range(4):
            np.testing.assert_array_equal(normals[i], [0, 0, 1])

    def test_dtypes(self) -> None:
        positions, normals, texcoords, indices = create_cube()
        assert positions.dtype == np.float32
        assert normals.dtype == np.float32
        assert texcoords.dtype == np.float32
        assert indices.dtype == np.uint32

    def test_uvs_in_unit_range(self) -> None:
        _, _, texcoords, _ = create_cube()
        assert texcoords.min() >= 0.0
        assert texcoords.max() <= 1.0


class TestCharucoTexture:
    """ChArUco checkerboard texture tests."""

    def test_output_shape(self) -> None:
        tex = create_charuco_texture(resolution=256)
        assert tex.shape == (256, 256, 3)

    def test_dtype(self) -> None:
        tex = create_charuco_texture()
        assert tex.dtype == np.uint8

    def test_contains_black_and_white(self) -> None:
        tex = create_charuco_texture()
        assert tex.min() == 0
        assert tex.max() == 255

    def test_checkerboard_pattern(self) -> None:
        tex = create_charuco_texture(squares_x=2, squares_y=2, resolution=4)
        # Top-left 2x2 block should be white (even+even)
        assert tex[0, 0, 0] == 255
        # Top-right 2x2 block should be black (even+odd)
        assert tex[0, 2, 0] == 0
