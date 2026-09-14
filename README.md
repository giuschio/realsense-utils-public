# realsense_utils2

A small set of utils for interacting with a RealSense D435 camera.

The package exposes three camera classes with the same capture API:

```python
color, depth = cam.capture()
```

- `SceneCameraRaw`: native RealSense depth aligned to the color frame.
- `SceneCameraCDM`: native RealSense depth aligned to color, then refined with
  `camera_depth_models`.
- `SceneCameraFDM`: FoundationStereo depth from the left/right IR pair, projected
  into the color frame.

`SceneCamera` is kept as a compatibility alias for `SceneCameraRaw`.

## Install
Tested with python 3.10 and 3.12. Neither the camera drivers nor the depth models have very stringent requirements, so they should work with most python, torch and Ubuntu version.


Clone the repository:

```bash
git clone https://github.com/giuschio/realsense-utils-public.git
cd realsense-utils-public
```

Install the camera package:

```bash
pip install -e ".[camera]"
```

With camera-depth-model integration:

```bash
git submodule update --init --recursive
pip install -e ./submodules/camera_depth_models
pip install -e ".[camera,depth-model]"
```

Then follow `submodules/camera_depth_models/README.md` to download the CDM
weights.

With FoundationStereo integration:

```bash
git submodule update --init --recursive
pip install -e ".[camera,foundation-stereo]"
```

Then follow `submodules/foundation_stereo/readme.md` to download weights into
`submodules/foundation_stereo/weights/`. The default expected checkpoint path is:

```text
submodules/foundation_stereo/weights/23-36-37/model_best_bp2_serialize.pth
```

## Basic Usage

```python
from realsense_utils2 import SceneCameraRaw

with SceneCameraRaw(resolution=(1280, 720)) as cam:
    color, depth = cam.capture()
    # color: uint8 BGR image, shape (H, W, 3)
    # depth: float32 depth map in meters, shape (H, W), aligned to color
    print(color.shape, color.dtype)
    print(depth.shape, depth.dtype)
```

RealSense RS400 visual presets can be passed to the camera classes:

```python
cam = SceneCameraRaw(visual_preset="high_density")
```

With CDM refinement:

```python
from realsense_utils2 import SceneCameraCDM

with SceneCameraCDM(camera_depth_model="/path/to/model.ckpt") as cam:
    color, depth = cam.capture()
```

With FoundationStereo:

```python
from realsense_utils2 import SceneCameraFDM

with SceneCameraFDM(
    foundation_depth_model="submodules/foundation_stereo/weights/23-36-37/model_best_bp2_serialize.pth",
    fdm_scale=0.5,
) as cam:
    color, depth = cam.capture()
```

## Demo

The demo captures one frame and displays a colored Open3D point cloud:

```bash
pip install -e ".[camera,visualization]"
```

```bash
python scripts/live_demo.py --camera raw
python scripts/live_demo.py --camera cdm --cdm-weights /path/to/model.ckpt
python scripts/live_demo.py --camera fdm
```

The demo displays only points within `2.0m` Euclidean camera-frame distance by
default and voxel downsamples at `0.004m` before opening the Open3D viewer.

`camera_depth_models` and FoundationStereo are intentionally not declared as
direct package dependencies. Install them through the included submodule
workflows above.
