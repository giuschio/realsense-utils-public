# realsense_utils2

`realsense_utils2` provides a lightweight RealSense camera class with optional depth-model integration.

## Install

Clone the repository:

```bash
git clone https://github.com/giuschio/realsense-utils-public.git
cd realsense-utils-public
```

Install the camera package:

```bash
pip install -e ".[camera]"
```

With camera-depth-model integration (submodule-based):

```bash
git submodule update --init --recursive
pip install -e ./submodules/camera_depth_models
pip install -e ".[camera,depth-model]"
```

Then follow:
`submodules/camera_depth_models/README.md`
to download/checkpoint the depth-model weights.

## Basic usage

```python
from realsense_utils2 import SceneCamera

with SceneCamera(resolution=(1280, 720)) as cam:
    color, depth = cam.capture()
    # color: uint8 BGR image, shape (H, W, 3)
    # depth: float32 depth map in meters, shape (H, W)
    print(color.shape, color.dtype)   # e.g. (720, 1280, 3) uint8
    print(depth.shape, depth.dtype)   # e.g. (720, 1280) float32
    print(depth.min(), depth.max())   # meters

    # Convert BGR -> RGB for plotting if needed:
    color_rgb = color[:, :, ::-1]
```

With depth-model refinement:

```python
cam = SceneCamera(camera_depth_model="/path/to/model.ckpt")
color, depth = cam.capture()
cam.finalize()
```

`camera_depth_models` is intentionally not declared as a direct package dependency in
`pyproject.toml`; install it through the included git submodule workflow above.
