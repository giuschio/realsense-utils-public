#!/usr/bin/env python3
"""Display RealSense color/depth frames; update on Enter."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from realsense_utils2 import SceneCamera


def depth_to_rgb(depth_m: np.ndarray) -> np.ndarray:
    """Convert metric depth map to a display-friendly RGB image."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.float32)

    lo = np.percentile(depth_m[valid], 2)
    hi = np.percentile(depth_m[valid], 98)
    if hi <= lo:
        hi = lo + 1e-6

    depth_norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    return plt.get_cmap("turbo")(depth_norm)[..., :3]



def main() -> None:

    with SceneCamera() as cam:
        fig, (ax_color, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title("realsense_utils2 example")
        ax_color.set_title("Color")
        ax_depth.set_title("Depth")
        ax_color.axis("off")
        ax_depth.axis("off")

        color_img = None
        depth_img = None

        print("Press Enter to capture/update. Type 'q' then Enter to quit.")
        while True:
            cmd = input("> ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break
            if cmd != "":
                print("Use Enter to update, or 'q' then Enter to quit.")
                continue

            color, depth = cam.capture()
            depth_rgb = depth_to_rgb(depth)

            if color_img is None:
                color_img = ax_color.imshow(color[:, :, ::-1])  # BGR -> RGB
                depth_img = ax_depth.imshow(depth_rgb)
            else:
                color_img.set_data(color[:, :, ::-1])
                depth_img.set_data(depth_rgb)

            fig.suptitle(
                f"Resolution: {cam.width}x{cam.height} | "
                f"Depth range: {np.nanmin(depth):.3f}m - {np.nanmax(depth):.3f}m"
            )
            plt.tight_layout()
            plt.pause(0.001)

        plt.close(fig)


if __name__ == "__main__":
    main()
