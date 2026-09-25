"""Depth reprojection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics with attribute and key access (``K.fx``, ``K["fx"]``)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __getitem__(self, key: str) -> float | int:
        """Return an intrinsic field by name; unknown keys raise ``KeyError``."""
        if key not in ("fx", "fy", "cx", "cy", "width", "height"):
            raise KeyError(key)
        return getattr(self, key)

    @classmethod
    def from_realsense(cls, intrinsics: Any) -> "CameraIntrinsics":
        return cls(
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
            width=int(intrinsics.width),
            height=int(intrinsics.height),
        )

    def scaled(self, scale: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=max(1, int(round(self.width * scale))),
            height=max(1, int(round(self.height * scale))),
        )

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from one camera frame to another."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_realsense(cls, extrinsics: Any) -> "CameraExtrinsics":
        return cls(
            rotation=np.asarray(extrinsics.rotation, dtype=np.float32).reshape(
                3, 3, order="F"
            ),
            translation=np.asarray(extrinsics.translation, dtype=np.float32).reshape(3),
        )


def reproject_depth_to_color_torch(
    depth_m: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    source_to_color: CameraExtrinsics,
    device: str = "cuda",
) -> np.ndarray:
    """Project source-frame depth into the color camera frame with z-buffering.

    Missing color pixels are returned as ``0.0``.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Torch is required for FDM color-frame reprojection."
        ) from exc

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    depth = torch.as_tensor(depth_m, dtype=torch.float32, device=device)
    height, width = depth.shape

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    valid = torch.isfinite(depth) & (depth > 0.0)
    if not torch.any(valid):
        return np.zeros(
            (color_intrinsics.height, color_intrinsics.width), dtype=np.float32
        )

    z = depth[valid]
    x = (xs[valid] - source_intrinsics.cx) * z / source_intrinsics.fx
    y = (ys[valid] - source_intrinsics.cy) * z / source_intrinsics.fy
    points = torch.stack((x, y, z), dim=0)

    rotation = torch.as_tensor(
        source_to_color.rotation, dtype=torch.float32, device=device
    )
    translation = torch.as_tensor(
        source_to_color.translation, dtype=torch.float32, device=device
    ).reshape(3, 1)
    color_points = rotation @ points + translation

    zc = color_points[2]
    in_front = zc > 0.0
    if not torch.any(in_front):
        return np.zeros(
            (color_intrinsics.height, color_intrinsics.width), dtype=np.float32
        )

    xc = color_points[0, in_front]
    yc = color_points[1, in_front]
    zc = zc[in_front]

    u = torch.round(color_intrinsics.fx * xc / zc + color_intrinsics.cx).long()
    v = torch.round(color_intrinsics.fy * yc / zc + color_intrinsics.cy).long()
    inside = (
        (u >= 0)
        & (u < color_intrinsics.width)
        & (v >= 0)
        & (v < color_intrinsics.height)
    )
    if not torch.any(inside):
        return np.zeros(
            (color_intrinsics.height, color_intrinsics.width), dtype=np.float32
        )

    linear = v[inside] * color_intrinsics.width + u[inside]
    zc = zc[inside]

    output = torch.full(
        (color_intrinsics.height * color_intrinsics.width,),
        torch.inf,
        dtype=torch.float32,
        device=device,
    )
    output.scatter_reduce_(0, linear, zc, reduce="amin", include_self=True)
    output = output.reshape(color_intrinsics.height, color_intrinsics.width)
    output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))
    return output.cpu().numpy().astype(np.float32, copy=False)


def source_depth_to_color_points(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    source_intrinsics: CameraIntrinsics,
    color_intrinsics: CameraIntrinsics,
    source_to_color: CameraExtrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert source-frame depth to color-frame points with sampled RGB colors."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
        )

    ys, xs = np.nonzero(valid)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - source_intrinsics.cx) * z / source_intrinsics.fx
    y = (ys.astype(np.float32) - source_intrinsics.cy) * z / source_intrinsics.fy
    source_points = np.stack((x, y, z), axis=0)

    color_points = (
        source_to_color.rotation @ source_points
        + source_to_color.translation.reshape(3, 1)
    )
    zc = color_points[2]
    in_front = zc > 0.0
    if not np.any(in_front):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
        )

    color_points = color_points[:, in_front]
    zc = zc[in_front]
    u = np.rint(color_intrinsics.fx * color_points[0] / zc + color_intrinsics.cx)
    v = np.rint(color_intrinsics.fy * color_points[1] / zc + color_intrinsics.cy)
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    inside = (
        (u >= 0)
        & (u < color_intrinsics.width)
        & (v >= 0)
        & (v < color_intrinsics.height)
    )
    if not np.any(inside):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
        )

    points = color_points[:, inside].T.astype(np.float32)
    colors_rgb = color_bgr[v[inside], u[inside]][:, ::-1].astype(np.float64) / 255.0
    return points, colors_rgb
