# Sparse Optical Flow Feature

This document describes the sparse optical flow feature in Lucidity Preprocessor, which enables efficient pre-calculation and storage of downsampled optical flow vectors within the circular mask region for surgical/endoscopic video processing.

## Overview

The optical flow feature uses the **RAFT (Recurrent All-Pairs Field Transforms)** model from PyTorch's torchvision library to compute dense motion estimation, then intelligently downsamples and filters to create a sparse vector field. This approach is optimised for:

- Surgical/endoscopic video where only the circular region matters
- Visualising motion patterns without storing full dense flow
- Extreme storage efficiency (100-1000x reduction vs dense flow)
- Fast vector field plotting and analysis

## Key Features

- **Sparse Sampling**: Downsamples flow field by configurable stride (8, 16, or 32 pixels)
- **Mask-Based Filtering**: Only stores vectors inside the circular endoscopic region
- **Automatic Mask Detection**: Detects circular mask from first frame
- **Efficient Storage**: Stores only vector coordinates and magnitudes
- **Easy Visualisation**: Built-in quiver plot and colour-coded plotting utilities
- **GPU Acceleration**: Automatically uses CUDA if available

## Storage Efficiency

Compared to dense optical flow:

| Resolution | Dense Flow | Sparse Flow (stride=16, masked) | Reduction |
|------------|------------|----------------------------------|-----------|
| 1920×1080 | ~16.6 MB/frame | ~20-50 KB/frame | 300-800x |
| 1280×720 | ~7.4 MB/frame | ~10-25 KB/frame | 300-700x |

The exact reduction depends on the mask size and stride.

## Installation

The optical flow feature requires:

```bash
pip install torch torchvision>=0.12.0 matplotlib
```

## Usage

### Command Line

Process a video with sparse optical flow:

```bash
python -m lucidity.cli process video.mp4 \
    --models raft_optical_flow \
    --output ./output
```

Process specific frame range:

```bash
python -m lucidity.cli process video.mp4 \
    --models raft_optical_flow \
    --start-frame 0 \
    --end-frame 100
```

### Python API

```python
from lucidity.plugin_manager import PluginManager
from lucidity.processor import VideoProcessor

# Create processor
processor = VideoProcessor(
    video_path="video.mp4",
    output_dir="./output",
)

# Add sparse optical flow model
processor.add_model("raft_optical_flow", config={
    "model_size": "small",      # or "large"
    "num_flow_updates": 12,     # RAFT iterations
    "stride": 16,               # Sample every 16 pixels
    "mask_threshold": 30,       # Black pixel threshold
    "mask_method": "hough",     # Circle detection method
})

# Process
manifest_path = processor.process()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | `"small"` | RAFT variant: `"small"` (faster) or `"large"` (more accurate) |
| `num_flow_updates` | int | `12` | Number of RAFT refinement iterations |
| `stride` | int | `16` | Sampling stride in pixels (8, 16, or 32 recommended) |
| `mask_threshold` | int | `30` | Black pixel threshold for mask detection (0-255) |
| `mask_method` | str | `"hough"` | Circle fitting method: `"hough"` or `"contour"` |

Example with custom configuration:

```python
processor.add_model("raft_optical_flow", config={
    "model_size": "small",
    "stride": 8,  # Denser sampling
    "mask_threshold": 40,
})
```

## Output Format

Sparse optical flow is saved as JSON (CUSTOM output type):

```
output/
└── raft_optical_flow/
    └── outputs.json
```

The JSON contains per-frame sparse flow data:

```json
[
  {
    "timestamp": 0.0333,
    "frame_number": 1,
    "data": {
      "x": [320.0, 336.0, 352.0, ...],  // X coordinates
      "y": [240.0, 240.0, 240.0, ...],  // Y coordinates
      "u": [1.2, -0.5, 0.8, ...],       // Horizontal flow
      "v": [0.3, 1.1, -0.2, ...]        // Vertical flow
    },
    "metadata": {
      "num_vectors": 543,
      "stride": 16,
      "mask_radius": 350.5,
      "mask_centre": [640.0, 480.0],
      "flow_magnitude_mean": 2.35,
      "flow_magnitude_max": 15.8,
      "flow_magnitude_std": 1.92
    }
  },
  ...
]
```

### Loading Sparse Flow Data

```python
import json
import numpy as np

# Load sparse flow
with open('output/raft_optical_flow/outputs.json', 'r') as f:
    flow_data = json.load(f)

# Get flow for specific frame
frame_10_flow = flow_data[10]['data']
x = np.array(frame_10_flow['x'])
y = np.array(frame_10_flow['y'])
u = np.array(frame_10_flow['u'])
v = np.array(frame_10_flow['v'])

# Calculate magnitudes
magnitudes = np.sqrt(u**2 + v**2)
```

## Visualisation

### Quick Visualisation with OpenCV

```python
from lucidity.flow_plotting import render_sparse_flow_on_frame
import cv2

# Read a frame
frame = cv2.imread('frame.png')

# Render flow vectors on frame
vis_frame = render_sparse_flow_on_frame(
    sparse_flow=frame_10_flow,
    frame=frame,
    scale=2.0,              # Arrow scale
    color=(0, 255, 255),    # Cyan arrows
    thickness=2,
)

# Save or display
cv2.imwrite('flow_vis.png', vis_frame)
```

### Quiver Plot with Matplotlib

```python
from lucidity.flow_plotting import plot_sparse_flow_quiver, save_sparse_flow_visualization
import matplotlib.pyplot as plt

# Create quiver plot
fig = plot_sparse_flow_quiver(
    sparse_flow=frame_10_flow,
    image_shape=(1080, 1920),
    background=frame,  # Optional background image
    scale=1.0,
    color='cyan',
    alpha=0.8,
)

plt.show()

# Or save directly
save_sparse_flow_visualization(
    sparse_flow=frame_10_flow,
    output_path='flow_quiver.png',
    image_shape=(1080, 1920),
    plot_type='quiver',
)
```

### Colour-Coded Visualisation

```python
from lucidity.flow_plotting import plot_sparse_flow_color

# Colour represents direction and magnitude
fig = plot_sparse_flow_color(
    sparse_flow=frame_10_flow,
    image_shape=(1080, 1920),
    background=frame,
    point_size=30,
)

plt.show()
```

## How It Works

1. **Dense Flow Computation**: RAFT computes dense optical flow for the entire frame
2. **Downsampling**: Flow field is sampled at regular grid points (every `stride` pixels)
3. **Mask Detection**: On first frame, detects circular endoscopic region
4. **Filtering**: Only vectors inside the mask are kept
5. **Storage**: Sparse vectors saved with their coordinates

### Example Numbers

For a 1920×1080 surgical video with stride=16:

- **Dense flow**: 1920 × 1080 × 2 = 4,147,200 values per frame
- **Downsampled grid**: 120 × 68 = 8,160 potential vectors
- **After masking** (circular region ~50% of frame): ~4,000 vectors
- **Storage reduction**: 4,147,200 / 4,000 = **1,036x smaller**

## Performance

Processing speed depends on GPU and resolution. The downsampling and masking steps add negligible overhead (<1ms per frame).

Typical speeds (RAFT Small on RTX 3090):
- 1920×1080: ~20-30 FPS
- 1280×720: ~40-50 FPS

## Advanced Usage

### Batch Processing Multiple Videos

```python
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']

for video_path in videos:
    processor = VideoProcessor(
        video_path=video_path,
        output_dir=f"./output/{Path(video_path).stem}",
    )
    processor.add_model("raft_optical_flow", config={"stride": 16})
    processor.process()
```

### Combining with Other Models

```python
# Process optical flow and depth simultaneously
processor.add_model("raft_optical_flow", config={"stride": 16})
processor.add_model("endomust_depth")

manifest_path = processor.process()
```

### Custom Stride for Different Analyses

```python
# Dense sampling for detailed motion analysis
config_dense = {"stride": 8}

# Coarse sampling for quick overview
config_coarse = {"stride": 32}
```

## Troubleshooting

### No Mask Detected

If mask detection fails:
- Adjust `mask_threshold` (try 20-50)
- Try different `mask_method` ('hough' or 'contour')
- Check that video contains a clear circular region

### Out of Memory

- Use `model_size: "small"`
- Reduce `num_flow_updates` to 8 or 6
- Process fewer frames at once

### Visualisation Issues

```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('Agg')  # For non-interactive environments
```

## References

- RAFT Paper: [https://arxiv.org/abs/2003.12039](https://arxiv.org/abs/2003.12039)
- PyTorch RAFT: [https://pytorch.org/vision/stable/models/optical_flow.html](https://pytorch.org/vision/stable/models/optical_flow.html)
- Sparse Flow Research: Sparse optical flow reduces storage while preserving motion patterns for analysis
