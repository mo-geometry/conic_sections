"""Camera model with intrinsic and extrinsic parameters.

Provides a dataclass-based camera representation that can generate
projection matrices suitable for both CPU-side ray tracing and GPU
shader uniforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from conic_sections.core.distortion import (
    DistortionCoeffs,
    DistortionLUT,
    LensModel,
    build_distortion_lut,
)


@dataclass
class Intrinsics:
    """Camera intrinsic parameters.

    Attributes:
        fx: Focal length in pixels (x-axis).
        fy: Focal length in pixels (y-axis).
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.
        dist_coeffs: Legacy radial/tangential distortion coefficients
            (k1, k2, p1, p2, k3).  Retained for OpenCV compatibility.
        lens_model: Fish-eye / wide-angle projection model.  Defaults to
            PINHOLE (no radial distortion).
        distortion_coeffs: Polynomial distortion coefficients (k2, k3, k4).
            Only used when ``lens_model`` is ``LensModel.POLYNOMIAL``.
    """

    fx: float = 800.0
    fy: float = 800.0
    cx: float = 320.0
    cy: float = 240.0
    dist_coeffs: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )
    lens_model: LensModel = LensModel.PINHOLE
    distortion_coeffs: DistortionCoeffs = field(default_factory=DistortionCoeffs)

    @property
    def distortion_lut(self) -> DistortionLUT:
        """Build (or return) the radial distortion look-up table.

        The LUT is regenerated each time this property is accessed.  For
        real-time rendering, cache the result externally and rebuild only
        when the lens model or coefficients change.
        """
        return build_distortion_lut(
            model=self.lens_model,
            coeffs=self.distortion_coeffs,
        )

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Return the 3x3 camera intrinsic matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass
class Extrinsics:
    """Camera extrinsic parameters (pose in world space).

    Attributes:
        position: Camera position in world coordinates [x, y, z].
        rotation: Rotation matrix (3x3, world-to-camera).
    """

    position: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    rotation: npt.NDArray[np.float64] = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    @property
    def view_matrix(self) -> npt.NDArray[np.float64]:
        """Return the 4x4 view matrix (world-to-camera transform)."""
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = -self.rotation @ self.position
        return mat


@dataclass
class Camera:
    """A complete camera model combining intrinsics and extrinsics.

    Supports generating OpenGL-compatible projection and view matrices
    for GPU rendering, as well as CPU-side ray generation for classical
    ray tracing.
    """

    intrinsics: Intrinsics = field(default_factory=Intrinsics)
    extrinsics: Extrinsics = field(default_factory=Extrinsics)
    width: int = 640
    height: int = 480
    near: float = 0.1
    far: float = 1000.0

    @property
    def projection_matrix(self) -> npt.NDArray[np.float64]:
        """Return a 4x4 OpenGL-style projection matrix derived from intrinsics."""
        k = self.intrinsics.matrix
        w, h = self.width, self.height
        n, f = self.near, self.far

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = 2.0 * k[0, 0] / w
        proj[1, 1] = 2.0 * k[1, 1] / h
        proj[0, 2] = 1.0 - 2.0 * k[0, 2] / w
        proj[1, 2] = 2.0 * k[1, 2] / h - 1.0
        proj[2, 2] = -(f + n) / (f - n)
        proj[2, 3] = -2.0 * f * n / (f - n)
        proj[3, 2] = -1.0
        return proj

    @property
    def view_matrix(self) -> npt.NDArray[np.float64]:
        """Return the 4x4 view matrix from extrinsics."""
        return self.extrinsics.view_matrix
