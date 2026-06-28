"""Radial lens distortion models for fish-eye and wide-angle cameras.

Provides forward (spherical ray → image plane) and inverse (image plane →
spherical ray) radial distortion using a look-up table (LUT) approach.
Five classical projection models are supported, plus a general polynomial.

The LUT maps field angle θ (radians, measured from the optical axis) to
image-plane radius r (normalised units — multiply by focal length to get
pixels).  Both directions use ``numpy.interp`` for fast vectorised lookup.

Coordinate convention (consistent with the rest of the repository):
- The optical axis is the +Z direction.
- Field angle θ = arccos(z) for a unit ray (x, y, z) on S².
- Azimuth φ = atan2(y, x) in the image plane.

Reference:
    O'Sullivan & Stec, IMVIP 2020 — fish-eye lens distortion modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class LensModel(Enum):
    """Supported radial distortion projection models.

    Each model defines the relationship r(θ) between the field angle θ
    (angle from the optical axis) and the image-plane radius r.

    Attributes:
        PINHOLE: Rectilinear projection — r = tan(θ). No distortion.
            Valid only for θ < π/2 (less than 90° half-angle).
        EQUIDISTANT: r = θ.  Linear mapping; the most common fish-eye model.
            Used by OpenCV ``cv2.fisheye``.
        EQUISOLID: r = 2·sin(θ/2).  Preserves solid angle (area on the sphere
            maps to proportional area on the sensor).
        STEREOGRAPHIC: r = 2·tan(θ/2).  Conformal — preserves local angles.
        ORTHOGRAPHIC: r = sin(θ).  Orthographic projection of the hemisphere.
            Clips at θ = π/2 (90° half-angle field of view).
        POLYNOMIAL: r = θ + k₂θ² + k₃θ³ + k₄θ⁴.  General calibration model.
            Coefficients k₂, k₃, k₄ are stored in ``DistortionCoeffs``.
    """

    PINHOLE = "Pinhole"
    EQUIDISTANT = "Equidistant"
    EQUISOLID = "Equisolid"
    STEREOGRAPHIC = "Stereographic"
    ORTHOGRAPHIC = "Orthographic"
    POLYNOMIAL = "Polynomial"


@dataclass
class DistortionCoeffs:
    """Polynomial distortion coefficients for the POLYNOMIAL lens model.

    The polynomial model is:  r(θ) = θ + k2·θ² + k3·θ³ + k4·θ⁴

    These coefficients are only used when ``LensModel.POLYNOMIAL`` is selected.
    For the other five models the r(θ) relationship is fully determined by
    the model name alone.

    Attributes:
        k2: Second-order coefficient.
        k3: Third-order coefficient.
        k4: Fourth-order coefficient.
    """

    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0


@dataclass
class DistortionLUT:
    """Pre-computed look-up table for radial distortion.

    Stores matched arrays of field angle (theta) and image-plane radius (r)
    that allow fast vectorised interpolation in both directions via
    ``numpy.interp``.

    Attributes:
        theta: Field angle values in radians, monotonically increasing.
        r: Corresponding image-plane radius values (normalised).
        model: The lens model used to generate this LUT.
    """

    theta: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    model: LensModel


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------


def build_distortion_lut(
    model: LensModel,
    coeffs: DistortionCoeffs | None = None,
    n_points: int = 4096,
) -> DistortionLUT:
    """Build a radial distortion look-up table for the given lens model.

    Generates *n_points* samples of θ from 0 to 0.9π and computes the
    corresponding image-plane radius r(θ) according to the selected model.

    For the POLYNOMIAL model the mapping is checked for monotonicity — a
    non-monotonic r(θ) means the polynomial coefficients are physically
    invalid (rays would cross in the image plane).

    Args:
        model: Which projection model to use.
        coeffs: Polynomial coefficients (required for POLYNOMIAL, ignored
            for other models).
        n_points: Number of LUT samples.  4096 gives sub-0.001° interpolation
            error across the full field.

    Returns:
        A ``DistortionLUT`` containing matched theta and r arrays.

    Raises:
        ValueError: If *model* is POLYNOMIAL and the resulting r(θ) is not
            monotonically increasing (invalid coefficients).
    """
    theta = np.linspace(0.0, 0.9 * np.pi, n_points, dtype=np.float64)

    if model == LensModel.PINHOLE:
        # Rectilinear: r = tan(θ).  Clip near π/2 to avoid infinity.
        r = np.tan(np.minimum(theta, np.pi / 2 - 1e-6))

    elif model == LensModel.EQUIDISTANT:
        r = theta.copy()

    elif model == LensModel.EQUISOLID:
        r = 2.0 * np.sin(theta / 2.0)

    elif model == LensModel.STEREOGRAPHIC:
        r = 2.0 * np.tan(theta / 2.0)

    elif model == LensModel.ORTHOGRAPHIC:
        # sin(θ) — clips at θ = π/2 (derivative goes to zero, then negative).
        # We follow the main-branch convention: accumulate |dr/dθ| and freeze
        # at the maximum, so the LUT stays monotonic.
        r_raw = np.cumsum(np.abs(np.gradient(np.sin(theta))))
        # Force r[0] = 0 (cumsum can introduce a small offset from gradient)
        r = np.clip(r_raw - r_raw[0], 0.0, 10.0)
        # Find the index closest to π/2 and freeze r and θ beyond it.
        clip_idx = int(np.argmin((theta - np.pi / 2.0) ** 2))
        r[clip_idx:] = r[clip_idx]
        theta[clip_idx:] = theta[clip_idx]

    elif model == LensModel.POLYNOMIAL:
        if coeffs is None:
            coeffs = DistortionCoeffs()
        r = theta + coeffs.k2 * theta**2 + coeffs.k3 * theta**3 + coeffs.k4 * theta**4
        # Monotonicity check — dr/dθ must be positive everywhere.
        if np.gradient(r).min() < 0:
            msg = (
                "Polynomial distortion is not monotonically increasing. "
                f"Coefficients k2={coeffs.k2}, k3={coeffs.k3}, k4={coeffs.k4} "
                "produce a non-physical lens model where rays cross."
            )
            raise ValueError(msg)

    else:
        msg = f"Unknown lens model: {model}"
        raise ValueError(msg)

    return DistortionLUT(theta=theta, r=r, model=model)


# ---------------------------------------------------------------------------
# Forward projection:  spherical rays  →  image-plane coordinates
# ---------------------------------------------------------------------------


def distort_points(
    xyz: npt.NDArray[np.float64],
    lut: DistortionLUT,
) -> npt.NDArray[np.float64]:
    """Project unit rays from the sphere onto the image plane via the lens model.

    Given N rays as (x, y, z) unit vectors on S², compute the corresponding
    image-plane coordinates (u, v, 1) in normalised units (before the camera
    matrix K is applied).

    The mapping is:
        1. θ = arccos(z)           — field angle from optical axis
        2. φ = atan2(y, x)         — azimuth angle
        3. r = interp(θ, LUT)      — image-plane radius from distortion model
        4. u = r·cos(φ),  v = r·sin(φ),  w = 1

    Args:
        xyz: Unit rays on S², shape (N, 3).
        lut: Pre-computed distortion look-up table.

    Returns:
        Image-plane coordinates, shape (N, 3) — columns are (u, v, 1).
    """
    # Field angle from the optical axis (+Z)
    field_angle = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))

    # Azimuth angle in the image plane
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])

    # Look up image-plane radius from the distortion model
    radius = np.interp(field_angle, lut.theta, lut.r)

    # Convert to Cartesian image-plane coordinates
    uv1 = np.empty((len(xyz), 3), dtype=np.float64)
    uv1[:, 0] = radius * np.cos(azimuth)
    uv1[:, 1] = radius * np.sin(azimuth)
    uv1[:, 2] = 1.0

    return uv1


# ---------------------------------------------------------------------------
# Inverse projection:  image-plane coordinates  →  spherical rays
# ---------------------------------------------------------------------------


def undistort_points(
    uv1: npt.NDArray[np.float64],
    lut: DistortionLUT,
) -> npt.NDArray[np.float64]:
    """Lift image-plane points back onto the unit sphere via the inverse lens model.

    Given N image-plane coordinates (u, v, 1) in normalised units (after
    the camera matrix K has been removed), compute the corresponding unit
    rays (x, y, z) on S².

    The mapping is:
        1. r = sqrt(u² + v²)       — image-plane radius
        2. φ = atan2(v, u)         — azimuth angle
        3. θ = interp(r, LUT⁻¹)   — field angle from inverse distortion model
        4. x = sin(θ)·cos(φ),  y = sin(θ)·sin(φ),  z = cos(θ)

    Args:
        uv1: Image-plane coordinates, shape (N, 3) — columns are (u, v, 1).
            The third column is ignored.
        lut: Pre-computed distortion look-up table.

    Returns:
        Unit rays on S², shape (N, 3).
    """
    # Image-plane radius
    r = np.sqrt(uv1[:, 0] ** 2 + uv1[:, 1] ** 2)

    # Azimuth angle — guard against division by zero at the optical centre
    safe_r = np.where(r > 1e-12, r, 1.0)
    cos_azi = np.where(r > 1e-12, uv1[:, 0] / safe_r, 1.0)
    sin_azi = np.where(r > 1e-12, uv1[:, 1] / safe_r, 0.0)

    # Inverse look-up: r → θ  (swap the LUT columns)
    field_angle = np.interp(r, lut.r, lut.theta)

    # Convert to unit ray on S²
    xyz = np.empty((len(uv1), 3), dtype=np.float64)
    xyz[:, 0] = np.sin(field_angle) * cos_azi
    xyz[:, 1] = np.sin(field_angle) * sin_azi
    xyz[:, 2] = np.cos(field_angle)

    return xyz


# ---------------------------------------------------------------------------
# Convenience: full projection pipeline helpers
# ---------------------------------------------------------------------------


def spherical_rays_to_pixel_coords(
    xyz: npt.NDArray[np.float64],
    lut: DistortionLUT,
    k_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Project unit rays through the distortion model and camera matrix to pixels.

    Combines ``distort_points`` with the camera matrix K:
        pixel = K @ distort(ray)

    Args:
        xyz: Unit rays on S², shape (N, 3).
        lut: Pre-computed distortion look-up table.
        k_matrix: 3x3 camera intrinsic matrix.

    Returns:
        Pixel coordinates, shape (N, 3) — columns are (px, py, 1).
    """
    uv1 = distort_points(xyz, lut)
    return uv1 @ k_matrix.T


def pixel_coords_to_spherical_rays(
    pixels: npt.NDArray[np.float64],
    lut: DistortionLUT,
    k_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Lift pixel coordinates back onto the unit sphere.

    Removes the camera matrix K then applies ``undistort_points``:
        ray = undistort(K⁻¹ @ pixel)

    Args:
        pixels: Pixel coordinates, shape (N, 3) — columns are (px, py, 1).
        lut: Pre-computed distortion look-up table.
        k_matrix: 3x3 camera intrinsic matrix.

    Returns:
        Unit rays on S², shape (N, 3).
    """
    k_inv = np.linalg.inv(k_matrix)
    uv1 = pixels @ k_inv.T
    return undistort_points(uv1, lut)
