"""Lightweight RealSense scene camera wrapper."""

from __future__ import annotations

import gc
from typing import Any, Optional

import numpy as np


def _clear_torch_cache() -> None:
    """Release Python references and return unused CUDA memory to the allocator."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ManagedDepthModel:
    """Thin lifecycle wrapper around the optional camera depth refinement model."""

    def __init__(self, model: Any, preferred_device: str) -> None:
        self.model = model
        self.preferred_device = preferred_device

    @property
    def device(self) -> str:
        return str(self.model.device)

    def infer_depth(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return self.model.infer_depth(color, depth)

    def park(self) -> None:
        self.model = self.model.to("cpu").eval()
        _clear_torch_cache()

    def use(self, device: Optional[str] = None) -> None:
        target_device = device or self.preferred_device
        self.model = self.model.to(target_device).eval()


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
        visual_preset: str = "high_density",
    ):
        """Initialize the camera pipeline and optional depth refinement model.

        Args:
            camera_depth_model: Optional checkpoint/path used by
                ``camera_depth_models.load_model``. If provided, raw depth is
                refined on each ``capture()`` call.
            resolution: Requested ``(width, height)`` for both color and depth streams.
            visual_preset: Name of the ``rs.rs400_visual_preset`` to apply to
                the depth sensor.
        """
        self.pipeline = None
        self.frame_align = None
        self.depth_scale = None
        self.depth_visual_preset = None
        self._color_intrinsics: Optional[dict[str, float]] = None
        self.cdm = None
        self._use_cdm = False
        self.resolution = resolution
        self.visual_preset = visual_preset

        self.initialize()

        if camera_depth_model is not None:
            self.cdm = self._load_camera_depth_model(camera_depth_model)
            self._use_cdm = True

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

    @property
    def color_intrinsics(self) -> Optional[dict[str, float]]:
        """Return cached color intrinsics as a dictionary."""
        return self._color_intrinsics

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return 3x3 color camera intrinsic matrix."""
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return np.array(
            [
                [self.color_intrinsics["fx"], 0.0, self.color_intrinsics["cx"]],
                [0.0, self.color_intrinsics["fy"], self.color_intrinsics["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

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
        model = load_model("vitl", model_path, device)
        return ManagedDepthModel(model=model, preferred_device=device)

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
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        if depth_sensor.supports(rs.option.visual_preset):
            try:
                preset = getattr(rs.rs400_visual_preset, self.visual_preset)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown RealSense RS400 visual preset: {self.visual_preset!r}"
                ) from exc
            try:
                preset_value = float(preset)
            except TypeError:
                if self.visual_preset != "high_density":
                    raise
                preset_value = 4.0
            depth_sensor.set_option(rs.option.visual_preset, preset_value)
            self.depth_visual_preset = depth_sensor.get_option(rs.option.visual_preset)

        color_intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._color_intrinsics = {
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
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
        if self.cdm is not None and self._use_cdm is True:
            depth = self.cdm.infer_depth(color, depth)
        return color, depth

    def finalize(self) -> None:
        """Stop the pipeline if running and release the handle."""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    def park_depth_model(self) -> None:
        """Move the optional depth model to CPU and release unused CUDA cache."""
        if self.cdm is not None:
            self.cdm.park()

    def load_depth_model(self, device: Optional[str] = None) -> None:
        """Move the optional depth model back to its preferred device."""
        if self.cdm is None:
            raise RuntimeError("No camera depth model is configured.")
        self.cdm.use(device=device)

    def set_use_depth_model(self, flag: bool) -> None:
        self._use_cdm = flag

    def __enter__(self) -> "SceneCamera":
        """Context-manager entry; returns this camera instance."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context-manager exit; always finalizes the camera pipeline."""
        self.finalize()
