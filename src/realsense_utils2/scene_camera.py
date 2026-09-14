"""RealSense scene camera wrappers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .foundation_stereo_model import (
    DEFAULT_FOUNDATION_STEREO_WEIGHTS,
    ManagedFoundationStereo,
)
from .projection import (
    CameraExtrinsics,
    CameraIntrinsics,
    reproject_depth_to_color_torch,
    source_depth_to_color_points,
)


DEFAULT_CAMERA_DEPTH_MODEL_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "submodules"
    / "camera_depth_models"
    / "weights"
    / "model.ckpt"
)


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


class SceneCameraBase:
    """Common context-manager and RealSense lifecycle helpers."""

    def __init__(
        self,
        resolution: tuple[int, int] = (1280, 720),
        visual_preset: Optional[str] = "high_density",
    ) -> None:
        self.pipeline = None
        self.resolution = resolution
        self.visual_preset = visual_preset
        self.depth_visual_preset: Optional[float] = None
        self._color_intrinsics: Optional[CameraIntrinsics] = None
        self.initialize()

    @property
    def width(self) -> int:
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return self.color_intrinsics.width

    @property
    def height(self) -> int:
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return self.color_intrinsics.height

    @property
    def color_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self._color_intrinsics

    @property
    def camera_matrix(self) -> np.ndarray:
        if self.color_intrinsics is None:
            raise RuntimeError("Camera is not initialized.")
        return self.color_intrinsics.matrix()

    def initialize(self) -> None:
        raise NotImplementedError

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _apply_visual_preset(self, rs: Any, depth_sensor: Any) -> None:
        """Apply an RS400 visual preset when the connected sensor supports it."""
        if self.visual_preset is None:
            return
        if not depth_sensor.supports(rs.option.visual_preset):
            return

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

    def finalize(self) -> None:
        """Stop the pipeline if running and release the handle."""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    def __enter__(self) -> "SceneCameraBase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()


class SceneCameraRaw(SceneCameraBase):
    """Color + native RealSense depth aligned into the color frame."""

    def __init__(
        self,
        resolution: tuple[int, int] = (1280, 720),
        visual_preset: Optional[str] = "high_density",
    ) -> None:
        self.frame_align = None
        self.depth_scale = None
        super().__init__(resolution=resolution, visual_preset=visual_preset)

    def initialize(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "SceneCameraRaw requires 'pyrealsense2'. Install with: "
                "pip install 'realsense_utils2[camera]'"
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
        self._apply_visual_preset(rs, depth_sensor)
        self._color_intrinsics = CameraIntrinsics.from_realsense(
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.pipeline is None
            or self.frame_align is None
            or self.depth_scale is None
        ):
            raise RuntimeError("Camera is not initialized.")

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.frame_align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned color/depth frames.")

        color = np.asanyarray(color_frame.get_data())
        depth = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * self.depth_scale
        )
        return np.copy(color), np.copy(depth)


class SceneCameraCDM(SceneCameraRaw):
    """Color + RealSense depth refined by camera_depth_models."""

    def __init__(
        self,
        camera_depth_model: str | Path = DEFAULT_CAMERA_DEPTH_MODEL_WEIGHTS,
        resolution: tuple[int, int] = (1280, 720),
        visual_preset: Optional[str] = "high_density",
    ) -> None:
        self.cdm: Optional[ManagedDepthModel] = None
        super().__init__(resolution=resolution, visual_preset=visual_preset)
        self.cdm = self._load_camera_depth_model(str(camera_depth_model))

    def _load_camera_depth_model(self, model_path: str) -> ManagedDepthModel:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "SceneCameraCDM requires 'torch'. Install with: "
                "pip install 'realsense_utils2[depth-model]'"
            ) from exc

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Camera depth model weights not found: {model_path}"
            )

        try:
            from camera_depth_models import load_model
        except ImportError as exc:
            raise ImportError(
                "SceneCameraCDM requires 'camera_depth_models'. Install with: "
                "git submodule update --init --recursive && "
                "pip install -e ./submodules/camera_depth_models"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model("vitl", model_path, device)
        return ManagedDepthModel(model=model, preferred_device=device)

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        color, depth = super().capture()
        if self.cdm is None:
            raise RuntimeError("No camera depth model is configured.")
        return color, self.cdm.infer_depth(color, depth)

    def park_depth_model(self) -> None:
        if self.cdm is not None:
            self.cdm.park()

    def use_depth_model(self, device: Optional[str] = None) -> None:
        if self.cdm is None:
            raise RuntimeError("No camera depth model is configured.")
        self.cdm.use(device=device)


class SceneCameraFDM(SceneCameraBase):
    """Color + FoundationStereo depth projected into the color frame."""

    def __init__(
        self,
        foundation_depth_model: str | Path = DEFAULT_FOUNDATION_STEREO_WEIGHTS,
        resolution: tuple[int, int] = (1280, 720),
        fdm_scale: float = 0.5,
        valid_iters: int = 8,
        max_disp: int = 192,
        device: Optional[str] = None,
        visual_preset: Optional[str] = "high_density",
    ) -> None:
        self.left_intrinsics: Optional[CameraIntrinsics] = None
        self.color_intrinsics_for_projection: Optional[CameraIntrinsics] = None
        self.left_to_color: Optional[CameraExtrinsics] = None
        self.baseline_m: Optional[float] = None
        self.fdm_scale = fdm_scale
        self.device = device
        self.fdm: Optional[ManagedFoundationStereo] = None
        super().__init__(resolution=resolution, visual_preset=visual_preset)
        try:
            self.fdm = ManagedFoundationStereo(
                model_path=foundation_depth_model,
                preferred_device=device,
                valid_iters=valid_iters,
                max_disp=max_disp,
                scale=fdm_scale,
            )
        except Exception:
            self.finalize()
            raise

    def initialize(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "SceneCameraFDM requires 'pyrealsense2'. Install with: "
                "pip install 'realsense_utils2[camera]'"
            ) from exc

        self.pipeline = rs.pipeline()
        config = rs.config()
        width, height = self.resolution
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, 30)
        profile = self.pipeline.start(config)
        self._apply_visual_preset(rs, profile.get_device().first_depth_sensor())

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        left_profile = profile.get_stream(
            rs.stream.infrared, 1
        ).as_video_stream_profile()
        right_profile = profile.get_stream(
            rs.stream.infrared, 2
        ).as_video_stream_profile()

        self._color_intrinsics = CameraIntrinsics.from_realsense(
            color_profile.get_intrinsics()
        )
        self.color_intrinsics_for_projection = self._color_intrinsics
        self.left_intrinsics = CameraIntrinsics.from_realsense(
            left_profile.get_intrinsics()
        )
        self.left_to_color = CameraExtrinsics.from_realsense(
            left_profile.get_extrinsics_to(color_profile)
        )
        left_to_right = CameraExtrinsics.from_realsense(
            left_profile.get_extrinsics_to(right_profile)
        )
        self.baseline_m = float(abs(left_to_right.translation[0]))
        if self.baseline_m <= 0.0:
            raise RuntimeError("Invalid RealSense stereo baseline.")

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.pipeline is None
            or self.left_intrinsics is None
            or self.color_intrinsics_for_projection is None
            or self.left_to_color is None
            or self.baseline_m is None
            or self.fdm is None
        ):
            raise RuntimeError("Camera is not initialized.")

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        if not color_frame or not left_frame or not right_frame:
            raise RuntimeError("Failed to capture color/IR stereo frames.")

        color = np.copy(np.asanyarray(color_frame.get_data()))
        left = np.copy(np.asanyarray(left_frame.get_data()))
        right = np.copy(np.asanyarray(right_frame.get_data()))

        scaled_left_intrinsics = self.left_intrinsics.scaled(self.fdm_scale)
        depth_left = self.fdm.infer_depth(
            left=left,
            right=right,
            fx=scaled_left_intrinsics.fx,
            baseline_m=self.baseline_m,
        )
        depth_color = reproject_depth_to_color_torch(
            depth_m=depth_left,
            source_intrinsics=scaled_left_intrinsics,
            color_intrinsics=self.color_intrinsics_for_projection,
            source_to_color=self.left_to_color,
            device=self.device or self.fdm.device,
        )
        return color, depth_color

    def capture_pointcloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture an FDM cloud directly, colorized from the RGB stream."""
        if (
            self.pipeline is None
            or self.left_intrinsics is None
            or self.color_intrinsics_for_projection is None
            or self.left_to_color is None
            or self.baseline_m is None
            or self.fdm is None
        ):
            raise RuntimeError("Camera is not initialized.")

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)
        if not color_frame or not left_frame or not right_frame:
            raise RuntimeError("Failed to capture color/IR stereo frames.")

        color = np.copy(np.asanyarray(color_frame.get_data()))
        left = np.copy(np.asanyarray(left_frame.get_data()))
        right = np.copy(np.asanyarray(right_frame.get_data()))

        scaled_left_intrinsics = self.left_intrinsics.scaled(self.fdm_scale)
        depth_left = self.fdm.infer_depth(
            left=left,
            right=right,
            fx=scaled_left_intrinsics.fx,
            baseline_m=self.baseline_m,
        )
        return source_depth_to_color_points(
            depth_m=depth_left,
            color_bgr=color,
            source_intrinsics=scaled_left_intrinsics,
            color_intrinsics=self.color_intrinsics_for_projection,
            source_to_color=self.left_to_color,
        )

    def park_depth_model(self) -> None:
        self.fdm.park()
        _clear_torch_cache()

    def use_depth_model(self, device: Optional[str] = None) -> None:
        self.fdm.use(device=device)


SceneCamera = SceneCameraRaw
