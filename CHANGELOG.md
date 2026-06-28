# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Scheme

- **Major (X)**: Breaking API changes, architectural rewrites
- **Minor (Y)**: New features, backwards-compatible additions
- **Patch (Z)**: Bug fixes, minor improvements, dependency updates

## [Unreleased]

### Added
- Radial lens distortion module (`core/distortion.py`) with five fish-eye projection
  models: Equidistant (r=θ), Equisolid (r=2·sin(θ/2)), Stereographic (r=2·tan(θ/2)),
  Orthographic (r=sin(θ)), Polynomial (r=θ+k₂θ²+k₃θ³+k₄θ⁴), plus Pinhole baseline
- LUT-based distortion/undistortion with 4096-point look-up table and `numpy.interp`
- Forward projection (`distort_points`): unit rays → image-plane coordinates
- Inverse projection (`undistort_points`): image-plane coordinates → unit rays
- Full pipeline helpers: `spherical_rays_to_pixel_coords`, `pixel_coords_to_spherical_rays`
- `LensModel` enum and `DistortionCoeffs` dataclass for polynomial coefficients
- `Intrinsics.lens_model` and `Intrinsics.distortion_coeffs` fields with `distortion_lut`
  property wiring distortion into the camera model
- Comprehensive test suite for distortion module: LUT generation, known analytical values,
  round-trip consistency, monotonicity checks, edge cases, full pipeline (30+ test cases)
- Sensor tilt compensation module (`core/sensor_tilt.py`) implementing IMVIP 2020 equations:
  `project_rays` (eqns 4a/4b, world→sensor) and `unproject_rays` (eqns 7a/7b, sensor→world)
- `TiltVector` dataclass and `tilt_from_angles(angle_deg, azimuth_deg)` constructor
- Sensor tilt test suite: tilt vector construction, identity, optical-centre fixed point,
  round-trip consistency, non-trivial behaviour, full distortion+tilt pipeline integration

## [0.3.0] - 2026-06-01

### Added
- Procedural camera mesh generator with box body, lens barrel, and viewfinder
- View frustum wireframe generator for visualising camera FOV
- SceneCamera class wrapping Camera model with world-space pose (position, Euler rotation)
- SceneCameraManager for adding, removing, selecting, and cycling scene cameras
- Viewport switching: toggle between orbit camera and any scene camera's perspective
- Dear ImGui integration with camera control panel (position/rotation sliders, frustum toggles)
- Per-vertex colour shader for camera mesh rendering (Blinn-Phong, no texture)
- Line shader with model matrix support for frustum wireframe rendering
- Keyboard shortcuts: Tab to toggle panel, 1-9 to select scene cameras, V to toggle view
- ImGui input capture: orbit camera ignores mouse when ImGui panels are active
- FPS-style controls when viewing through a scene camera:
  mouse yaw/pitch, left-click drag roll, scroll zoom (FOV), WASD/arrow horizontal
  movement, Space/Shift+Space for ascend/descend, Shift for double speed,
  double left-click to reset horizon
- Ctrl toggle to freeze/unfreeze camera orientation (frees cursor for ImGui)
- GLFW cursor capture (disabled cursor) when in scene camera FPS mode
- World coordinate axes visualisation (+X red, +Y green, +Z blue) with ImGui toggle
- FOV slider in ImGui camera panel
- Horizontal-plane movement: WASD/arrows project onto XZ plane regardless of pitch
- Pyright/Pylance configuration in pyproject.toml for third-party library compatibility
- Tests for camera mesh geometry (13 tests), scene camera (17 tests), FPS controls (9 tests),
  world axes (4 tests)
- imgui-bundle[glfw] and PyOpenGL added to project dependencies

### Changed
- Camera coordinate convention: camera now looks along +Z (was -Z), +Y up, +X right
- Frustum wireframe extends along +Z to match new camera convention
- Euler angle labels in ImGui panel corrected to Pitch/Yaw/Roll
- Escape key exits scene camera view before quitting application
- Frustum wireframe regenerated per-frame to track FOV changes from scroll zoom

### Fixed
- View matrix reflection: negating both X and Z rows (det=+1) matches glm::lookAt
  convention, fixing left-right mirroring when viewing through scene cameras
- Scene camera textures: correct winding order eliminates face culling of textured
  geometry when viewed through scene cameras
- macOS OpenGL core profile: removed glLineWidth >1.0 calls that caused GL_INVALID_VALUE

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

[Unreleased]: https://github.com/mo-geometry/conic_sections/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mo-geometry/conic_sections/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mo-geometry/conic_sections/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mo-geometry/conic_sections/releases/tag/v0.1.0
