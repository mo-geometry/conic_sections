# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Scheme

- **Major (X)**: Breaking API changes, architectural rewrites
- **Minor (Y)**: New features, backwards-compatible additions
- **Patch (Z)**: Bug fixes, minor improvements, dependency updates

## [Unreleased]

## [0.2.0] - 2026-05-24

### Added
- ModernGL proof-of-concept renderer with GLFW windowed application
- Procedural cube mesh generator with per-face normals and UV coordinates
- ChArUco checkerboard texture generator for calibration targets
- Blinn-Phong vertex/fragment shaders with texture and solid-colour support
- Ground-plane grid with colour-coded axes (red X, blue Z)
- Mouse-driven orbit camera controller (drag to rotate, scroll to zoom)
- Tests for cube mesh geometry, texture generation, and orbit camera (12 tests)
- Session 1 homework sheet (GPU pipeline foundations)

### Fixed
- Projection matrix row/column-major transpose for OpenGL upload
- Model matrix translation placement (row-major convention)

## [0.1.0] - 2026-05-24

### Added
- Project scaffolding: src layout, pyproject.toml with hatchling build backend
- Camera model with intrinsic/extrinsic parameters and OpenGL projection matrices
- Geometric transforms: Euler-to-rotation, axis-angle (Rodrigues), look-at view matrix
- Headless ModernGL context helper for off-screen rendering and CI
- GitHub Actions CI/CD: lint (ruff), typecheck (mypy strict), test matrix (3.11/3.12/3.13)
- Pre-commit hooks: ruff lint + format, mypy, trailing whitespace, YAML/TOML checks
- Makefile with dev, lint, format, typecheck, test, clean targets
- Test suite with 21 passing tests covering camera math and transforms
- README, CONTRIBUTING guide, .gitignore with asset and IDE exclusions

[Unreleased]: https://github.com/mo-geometry/conic_sections/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mo-geometry/conic_sections/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mo-geometry/conic_sections/releases/tag/v0.1.0
