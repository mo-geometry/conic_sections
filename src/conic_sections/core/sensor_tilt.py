"""Sensor tilt compensation for off-axis lens mounting.

Models the geometric effect of a sensor plane that is tilted relative to
the optical axis.  When the sensor is not perfectly perpendicular to the
lens axis, image-plane coordinates are warped by a projective (homography-
like) transformation that depends on the tilt direction and magnitude.

Two operations are provided:

- **project_rays** (world → sensor, IMVIP eqns 4a & 4b):
  Given normalised image-plane coordinates produced by the lens distortion
  model, compute where those points land on the tilted sensor.

- **unproject_rays** (sensor → world, IMVIP eqns 7a & 7b):
  Given coordinates measured on the tilted sensor, remove the tilt effect
  to recover the coordinates that would have been observed on an untilted
  sensor.

The tilt is parameterised by a unit normal vector **n** = (nx, ny, nz)
that describes the sensor-plane orientation.  A convenience constructor
builds **n** from a tilt angle (how far the sensor is tilted) and an
azimuth (the direction of the tilt in the sensor plane).

Coordinate convention (consistent with the rest of the repository):
- The optical axis is the +Z direction.
- An untilted sensor has n = (0, 0, −1)  (normal points back toward lens).
- Tilt angle θ = 0 means no tilt; the equations reduce to the identity.

Reference:
    O'Sullivan & Stec, "Accurate Modelling of Fish-Eye Lens Distortion",
    IMVIP 2020, eqns 4a/4b (world→sensor) and 7a/7b (sensor→world).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class TiltVector:
    """Sensor-plane normal vector for the tilt model.

    Attributes:
        nx: X-component of the tilt normal.
        ny: Y-component of the tilt normal.
        nz: Z-component of the tilt normal (−1 when untilted).
    """

    nx: float = 0.0
    ny: float = 0.0
    nz: float = -1.0

    @property
    def is_identity(self) -> bool:
        """Return True when the tilt has no effect (sensor is perpendicular).

        When |nz| == 1 the sensor is perfectly aligned with the optical
        axis and both project/unproject reduce to the identity transform.
        """
        return abs(self.nz) == 1.0


def tilt_from_angles(
    tilt_angle_deg: float,
    tilt_azimuth_deg: float,
) -> TiltVector:
    """Construct a tilt vector from angle and azimuth in degrees.

    Args:
        tilt_angle_deg: Tilt magnitude in degrees.  0 means no tilt.
            Positive values tilt the sensor away from perpendicular.
        tilt_azimuth_deg: Direction of the tilt in the sensor plane,
            measured counter-clockwise from the +X axis, in degrees.

    Returns:
        A ``TiltVector`` encoding the sensor-plane normal.
    """
    theta = np.radians(tilt_angle_deg)
    azimuth = np.radians(tilt_azimuth_deg)
    return TiltVector(
        nx=float(np.sin(theta) * np.cos(azimuth)),
        ny=float(np.sin(theta) * np.sin(azimuth)),
        nz=float(-np.cos(theta)),
    )


# ---------------------------------------------------------------------------
# IMVIP eqns 4a & 4b  —  world → sensor  (project_rays)
# ---------------------------------------------------------------------------


def project_rays(
    uv1: npt.NDArray[np.float64],
    tilt: TiltVector,
) -> npt.NDArray[np.float64]:
    """Apply sensor tilt: world image-plane → tilted sensor coordinates.

    Implements IMVIP equations 4a and 4b.  Takes normalised image-plane
    coordinates (after radial distortion, before the camera matrix K) and
    returns the coordinates as they appear on the tilted sensor.

    Args:
        uv1: Normalised image-plane coordinates, shape (N, 3).
            Columns are (u, v, 1).
        tilt: Sensor-plane normal vector.

    Returns:
        Tilted sensor coordinates, shape (N, 3) — columns (u', v', 1).
    """
    if tilt.is_identity:
        return uv1.copy()

    nx, ny, nz = tilt.nx, tilt.ny, tilt.nz
    px = uv1[:, 0]
    py = uv1[:, 1]

    result = uv1.copy()
    denom = nx * px + ny * py + nz
    nz_m1 = nz - 1.0

    # eqn 4a
    result[:, 0] = ((nx**2 + nz * nz_m1) * px + nx * ny * py) / (denom * nz_m1)
    # eqn 4b
    result[:, 1] = ((ny**2 + nz * nz_m1) * py + nx * ny * px) / (denom * nz_m1)

    return result


# ---------------------------------------------------------------------------
# IMVIP eqns 7a & 7b  —  sensor → world  (unproject_rays)
# ---------------------------------------------------------------------------


def unproject_rays(
    uv1: npt.NDArray[np.float64],
    tilt: TiltVector,
) -> npt.NDArray[np.float64]:
    """Remove sensor tilt: tilted sensor coordinates → world image-plane.

    Implements IMVIP equations 7a and 7b.  Takes coordinates measured on
    the tilted sensor (after removing the camera matrix K) and recovers
    the normalised image-plane coordinates that would have been observed
    on an untilted sensor.

    Args:
        uv1: Tilted sensor coordinates, shape (N, 3).
            Columns are (u', v', 1).
        tilt: Sensor-plane normal vector.

    Returns:
        Untilted image-plane coordinates, shape (N, 3) — columns (u, v, 1).
    """
    if tilt.is_identity:
        return uv1.copy()

    nx, ny, nz = tilt.nx, tilt.ny, tilt.nz
    px = uv1[:, 0]
    py = uv1[:, 1]

    result = uv1.copy()
    denom = nx * px + ny * py + 1.0
    nz_m1 = nz - 1.0

    # eqn 7a
    result[:, 0] = ((nx**2 + nz_m1) * px + nx * ny * py) / (denom * nz_m1)
    # eqn 7b
    result[:, 1] = ((ny**2 + nz_m1) * py + nx * ny * px) / (denom * nz_m1)

    return result
