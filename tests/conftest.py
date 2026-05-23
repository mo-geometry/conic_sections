"""Shared test fixtures for the conic_sections test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import pytest

from conic_sections.core.camera import Camera, Extrinsics, Intrinsics


@pytest.fixture  # type: ignore[misc]
def default_camera() -> Iterator[Camera]:
    """A camera at the origin with default intrinsics."""
    yield Camera()


@pytest.fixture  # type: ignore[misc]
def translated_camera() -> Iterator[Camera]:
    """A camera offset from the origin along the z-axis."""
    yield Camera(
        extrinsics=Extrinsics(
            position=np.array([0.0, 0.0, 5.0]),
        ),
    )


@pytest.fixture  # type: ignore[misc]
def custom_intrinsics() -> Iterator[Intrinsics]:
    """Non-default intrinsics for testing projection math."""
    yield Intrinsics(fx=1000.0, fy=1000.0, cx=512.0, cy=384.0)
