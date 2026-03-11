"""Lightweight RealSense scene camera wrapper."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class SceneCamera:
    """Simple color+depth capture helper for Intel RealSense devices.

    Notes:
        Color frames are configured as ``rs.format.bgr8`` and returned in BGR order.
        Depth frames are returned in meters as ``float32``.
    """

    def __init__(
        self,
        camera_depth_model: Optional[str] = None,
        resolution: tuple[int, int] = (1280, 720),
    ):
        """Initialize the camera pipeline and optional depth refinement model.

        Args:
            camera_depth_model: Optional checkpoint/path used by
                ``camera_depth_models.load_model``. If provided, raw depth is
                refined on each ``capture()`` call.
            resolution: Requested ``(width, height)`` for both color and depth streams.
        """
        self.pipeline = None
        self.frame_align = None
        self.depth_scale = None
        self.color_intrinsics: Optional[dict[str, float]] = None
        self.cdm = None
        self.resolution = resolution

        self.initialize()

        if camera_depth_model is not None:
            self.cdm = self._load_camera_depth_model(camera_depth_model)

    @property
    def width(self) -> int:
        """Return active color stream width in pixels."""
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return int(self.color_intrinsics["width"])

    @property
    def height(self) -> int:
        """Return active color stream height in pixels."""
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return int(self.color_intrinsics["height"])

    def _load_camera_depth_model(self, model_path: str) -> Any:
        """Load and return the optional camera depth refinement model."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "camera_depth_model requires 'torch'. Install with: "
                "pip install 'realsense_utils2[depth-model]'"
            ) from exc

        try:
            from camera_depth_models import load_model
        except ImportError as exc:
            raise ImportError(
                "camera_depth_model requires 'camera_depth_models'. Install with: "
                "git submodule update --init --recursive && "
                "pip install -e ./submodules/camera_depth_models "
                "(then follow the submodule README for checkpoints)."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return load_model("vitl", model_path, device)

    def initialize(self) -> None:
        """Start RealSense streams and cache alignment, scale, and intrinsics."""
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "SceneCamera requires 'pyrealsense2'. Install it, or install package "
                "camera dependencies when available for your platform."
            ) from exc

        self.pipeline = rs.pipeline()
        config = rs.config()
        width, height = self.resolution
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.frame_align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        color_sensor = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.color_intrinsics = {
            "fx": color_sensor.fx,
            "fy": color_sensor.fy,
            "cx": color_sensor.ppx,
            "cy": color_sensor.ppy,
            "width": color_sensor.width,
            "height": color_sensor.height,
        }

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture one aligned color/depth pair.

        Returns:
            A tuple ``(color, depth)`` where:
            - ``color`` is a ``uint8`` BGR image with shape ``(H, W, 3)``.
            - ``depth`` is a ``float32`` depth map in meters with shape ``(H, W)``.
        """
        if self.pipeline is None or self.frame_align is None or self.depth_scale is None:
            raise RuntimeError("Camera is not initialized.")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.frame_align.process(frames)

        color = np.asanyarray(aligned_frames.get_color_frame().get_data())
        depth = (
            np.asanyarray(aligned_frames.get_depth_frame().get_data()).astype(np.float32)
            * self.depth_scale
        )

        # Return detached arrays, so upstream operations can't mutate frame-backed memory.
        color, depth = np.copy(color), np.copy(depth)
        if self.cdm is not None:
            depth = self.cdm.infer_depth(color, depth)
        return color, depth

    def finalize(self) -> None:
        """Stop the pipeline if running and release the handle."""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    def __enter__(self) -> "SceneCamera":
        """Context-manager entry; returns this camera instance."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context-manager exit; always finalizes the camera pipeline."""
        self.finalize()
