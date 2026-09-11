"""Runtime wrapper for the FoundationStereo submodule."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FOUNDATION_STEREO_ROOT = REPO_ROOT / "submodules" / "foundation_stereo"
DEFAULT_FOUNDATION_STEREO_WEIGHTS = (
    FOUNDATION_STEREO_ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
)


class ManagedFoundationStereo:
    """Thin lifecycle wrapper around the optional FoundationStereo model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_FOUNDATION_STEREO_WEIGHTS,
        preferred_device: Optional[str] = None,
        valid_iters: int = 8,
        max_disp: int = 192,
        scale: float = 0.5,
        hiera: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.valid_iters = valid_iters
        self.max_disp = max_disp
        self.scale = scale
        self.hiera = hiera

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "SceneCameraFDM requires 'torch'. Install FoundationStereo "
                "dependencies and this package's depth-model extra."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FoundationStereo weights not found: {self.model_path}. "
                "Download weights into submodules/foundation_stereo/weights/."
            )

        if str(FOUNDATION_STEREO_ROOT) not in sys.path:
            sys.path.insert(0, str(FOUNDATION_STEREO_ROOT))

        try:
            import cv2
            import yaml
            from omegaconf import OmegaConf
            from Utils import AMP_DTYPE
            from core.utils.utils import InputPadder
        except ImportError as exc:
            raise ImportError(
                "SceneCameraFDM requires FoundationStereo dependencies. Install with: "
                "pip install -r submodules/foundation_stereo/requirements.txt"
            ) from exc

        cfg_path = self.model_path.parent / "cfg.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"FoundationStereo config not found: {cfg_path}")

        with cfg_path.open("r") as cfg_file:
            cfg: dict[str, Any] = yaml.safe_load(cfg_file)
        cfg["valid_iters"] = valid_iters
        cfg["max_disp"] = max_disp
        cfg["scale"] = scale
        cfg["hiera"] = int(hiera)
        self.args = OmegaConf.create(cfg)

        self.torch = torch
        self.cv2 = cv2
        self.amp_dtype = AMP_DTYPE
        self.input_padder_type = InputPadder
        self.preferred_device = preferred_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.autograd.set_grad_enabled(False)
        self.model = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.use()

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    def park(self) -> None:
        self.model = self.model.to("cpu").eval()

    def use(self, device: Optional[str] = None) -> None:
        target_device = device or self.preferred_device
        self.model = self.model.to(target_device).eval()

    def infer_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Infer left-to-right disparity from a rectified stereo pair."""
        torch = self.torch
        cv2 = self.cv2
        device = self.device

        left_3c = self._ensure_three_channels(left)
        right_3c = self._ensure_three_channels(right)

        if self.scale != 1.0:
            left_3c = cv2.resize(left_3c, dsize=None, fx=self.scale, fy=self.scale)
            right_3c = cv2.resize(
                right_3c, dsize=(left_3c.shape[1], left_3c.shape[0])
            )

        height, width = left_3c.shape[:2]
        left_tensor = (
            torch.as_tensor(left_3c, device=device).float()[None].permute(0, 3, 1, 2)
        )
        right_tensor = (
            torch.as_tensor(right_3c, device=device).float()[None].permute(0, 3, 1, 2)
        )
        padder = self.input_padder_type(
            left_tensor.shape, divis_by=32, force_square=False
        )
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

        autocast_enabled = str(device).startswith("cuda")
        with torch.amp.autocast(
            "cuda", enabled=autocast_enabled, dtype=self.amp_dtype
        ):
            if self.hiera:
                disp = self.model.run_hierachical(
                    left_tensor,
                    right_tensor,
                    iters=self.valid_iters,
                    test_mode=True,
                    small_ratio=0.5,
                )
            else:
                disp = self.model.forward(
                    left_tensor,
                    right_tensor,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )

        disp = padder.unpad(disp.float())
        return disp.detach().cpu().numpy().reshape(height, width).clip(0, None)

    def infer_depth(
        self, left: np.ndarray, right: np.ndarray, fx: float, baseline_m: float
    ) -> np.ndarray:
        """Infer metric depth in the left image frame."""
        disp = self.infer_disparity(left, right)
        depth = np.zeros_like(disp, dtype=np.float32)
        valid = np.isfinite(disp) & (disp > 0.0)
        depth[valid] = np.float32(fx * baseline_m) / disp[valid]
        return depth

    @staticmethod
    def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=2)
        if image.ndim == 3 and image.shape[2] >= 3:
            return image[..., :3]
        raise ValueError(f"Expected HxW or HxWx3 image, got shape {image.shape}.")
