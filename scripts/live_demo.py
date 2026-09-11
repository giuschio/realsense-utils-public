#!/usr/bin/env python3
"""Capture one RealSense frame and show a colored Open3D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from realsense_utils2 import SceneCameraCDM, SceneCameraFDM, SceneCameraRaw
from realsense_utils2.foundation_stereo_model import DEFAULT_FOUNDATION_STEREO_WEIGHTS
from realsense_utils2.scene_camera import DEFAULT_CAMERA_DEPTH_MODEL_WEIGHTS


FDM_SCALE = 0.5
FDM_VALID_ITERS = 8
FDM_MAX_DISP = 192


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT resolution strings."""
    try:
        width, height = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "resolution must be formatted as WIDTHxHEIGHT, e.g. 1280x720"
        ) from exc


def make_camera(args: argparse.Namespace):
    if args.camera == "raw":
        return SceneCameraRaw(resolution=args.resolution)
    if args.camera == "cdm":
        return SceneCameraCDM(
            camera_depth_model=args.cdm_weights,
            resolution=args.resolution,
        )
    if args.camera == "fdm":
        return SceneCameraFDM(
            foundation_depth_model=args.fdm_weights,
            resolution=args.resolution,
            fdm_scale=FDM_SCALE,
            valid_iters=FDM_VALID_ITERS,
            max_disp=FDM_MAX_DISP,
        )
    raise ValueError(f"Unsupported camera mode: {args.camera}")


def depth_to_colored_points(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project color-frame depth and filter by Euclidean distance."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid):
        return empty_cloud()

    ys, xs = np.nonzero(valid)
    z = depth_m[ys, xs]
    x = (xs.astype(np.float32) - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (ys.astype(np.float32) - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    points = np.stack((x, y, z), axis=1).astype(np.float32)

    if max_distance > 0:
        keep = np.linalg.norm(points, axis=1) <= max_distance
        points = points[keep]
        ys = ys[keep]
        xs = xs[keep]

    colors_rgb = color_bgr[ys, xs][:, ::-1].astype(np.float64) / 255.0
    return points, colors_rgb


def empty_cloud() -> tuple[np.ndarray, np.ndarray]:
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float64)


def show_pointcloud(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    voxel_size: float,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Point-cloud view requires open3d. Install with: pip install open3d"
        ) from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

    if len(cloud.points) == 0:
        return

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(
        window_name="realsense_utils2 pointcloud",
        width=1280,
        height=900,
    ):
        raise RuntimeError("Open3D failed to create a visualization window.")

    try:
        vis.add_geometry(cloud)
        vis.add_geometry(axes)
        points_np = np.asarray(cloud.points)
        center = np.median(points_np, axis=0)
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_lookat(center)
        ctr.set_zoom(0.7)

        vis.run()
    finally:
        vis.destroy_window()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        choices=("raw", "cdm", "fdm"),
        default="fdm",
        help="Depth source to use.",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(1280, 720),
        help="Camera stream resolution as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--cdm-weights",
        type=Path,
        default=DEFAULT_CAMERA_DEPTH_MODEL_WEIGHTS,
        help="camera_depth_models checkpoint path.",
    )
    parser.add_argument(
        "--fdm-weights",
        type=Path,
        default=DEFAULT_FOUNDATION_STEREO_WEIGHTS,
        help="FoundationStereo checkpoint path.",
    )
    parser.add_argument(
        "--voxel-distance",
        type=float,
        default=2.0,
        help="Maximum displayed Euclidean camera-frame distance in meters.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.002,
        help="Voxel size in meters for point-cloud downsampling.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with make_camera(args) as cam:
        color, depth = cam.capture()
        points, colors_rgb = depth_to_colored_points(
            color_bgr=color,
            depth_m=depth,
            camera_matrix=cam.camera_matrix,
            max_distance=args.voxel_distance,
        )
    show_pointcloud(points, colors_rgb, voxel_size=args.voxel_size)


if __name__ == "__main__":
    main()
