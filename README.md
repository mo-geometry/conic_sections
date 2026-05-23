# Conic Sections

GPU-accelerated 3D camera simulation, ray tracing, and virtual environment
navigation built with ModernGL and GLFW.

## Features (planned)

- Camera intrinsic/extrinsic modelling with OpenGL-compatible projection matrices
- Blinn-Phong shading via GLSL vertex/fragment shaders
- Blender asset pipeline (glTF 2.0 import)
- Multi-camera rigs for drone/vehicle simulation
- Virtual environment navigation

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
make dev
make test
```

## References

- O'Sullivan & Stec, *IMVIP 2020* — fish-eye lens distortion modelling
- US Patent — compensating for off-axis lens tilt
