# Optical Flow Feature

This document describes the optical flow feature in Lucidity Preprocessor, which enables efficient pre-calculation and storage of dense optical flow for video processing.

## Overview

The optical flow feature uses the **RAFT (Recurrent All-Pairs Field Transforms)** model from PyTorch's torchvision library to compute dense motion estimation between consecutive video frames. This is particularly useful for:

- Motion analysis in surgical/endoscopic video
- Tracking instrument movement
- Scene motion characterisation
- Pre-processing for downstream ML models

## Features

- **Efficient Model**: Uses RAFT Small by default for fast processing
- **Compressed Storage**: Saves optical flow in compressed .npz format with typical 10-20x compression
- **GPU Acceleration**: Automatically uses CUDA if available
- **Flow Visualisation**: Generates RGB visualisations of flow fields using HSV colour wheel
- **Flexible Configuration**: Configurable model size and processing parameters

## Installation

The optical flow feature requires:

```bash
pip install torch torchvision>=0.12.0
```

## Usage

### Command Line

Process a video with optical flow using compressed output format:

```bash
python -m lucidity.cli process video.mp4 \
    --models raft_optical_flow \
    --output ./output \
    --output-format compressed
```

Process specific frame range:

```bash
python -m lucidity.cli process video.mp4 \
    --models raft_optical_flow \
    --start-frame 0 \
    --end-frame 100 \
    --output-format compressed
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

# Add optical flow model
processor.add_model("raft_optical_flow", config={
    "model_size": "small",      # or "large"
    "num_flow_updates": 12,     # RAFT iterations (12 is default)
})

# Process with compressed output
manifest_path = processor.process(
    output_format='compressed',  # Efficient storage
)
```

## Configuration Options

The optical flow model accepts the following configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | `"small"` | Model variant: `"small"` (faster) or `"large"` (more accurate) |
| `num_flow_updates` | int | `12` | Number of RAFT refinement iterations |
| `weights` | str | `"default"` | Path to custom weights, or `"default"` for pretrained |

Example with custom configuration:

```python
processor.add_model("raft_optical_flow", config={
    "model_size": "large",
    "num_flow_updates": 20,
})
```

## Output Format

### Compressed Format (Recommended)

When using `--output-format compressed`, optical flow is saved as a single `.npz` file:

```
output/
└── raft_optical_flow/
    └── frames.npz
```

The `.npz` file contains:
- `frames`: Optical flow fields (N, H, W, 2) - u, v components
- `timestamps`: Frame timestamps (N,)
- `frame_numbers`: Frame numbers (N,)

Load the data:

```python
import numpy as np

# Load optical flow
with np.load('output/raft_optical_flow/frames.npz') as data:
    flows = data['frames']           # (N, H, W, 2)
    timestamps = data['timestamps']  # (N,)
    frame_numbers = data['frame_numbers']  # (N,)

# Get flow for specific frame
flow_frame_10 = flows[10]  # (H, W, 2)
u_component = flow_frame_10[..., 0]  # Horizontal flow
v_component = flow_frame_10[..., 1]  # Vertical flow
```

### Individual Frames Format

When using `--output-format frames`, each flow field is saved separately:

```
output/
└── raft_optical_flow/
    └── frames/
        ├── frame_000001.npy
        ├── frame_000002.npy
        └── ...
```

Note: Frame 0 is skipped (no previous frame for comparison).

## Storage Efficiency

Optical flow data compresses very well:

| Resolution | Uncompressed Size | Compressed Size | Ratio |
|------------|-------------------|-----------------|-------|
| 1920×1080 | ~16.6 MB/frame | ~1-2 MB/frame | 10-15x |
| 1280×720 | ~7.4 MB/frame | ~0.5-1 MB/frame | 10-15x |
| 640×480 | ~2.5 MB/frame | ~0.2-0.3 MB/frame | 10-12x |

**Recommendation**: Always use `--output-format compressed` for optical flow to save disk space.

## Visualisation

The model automatically generates RGB visualisations of the flow field, stored in the metadata. Access them via:

```python
from lucidity.manifest import load_manifest

manifest = load_manifest('output/manifest.json')

# Get timeline entries
timeline = processor.timeline.get_model_timeline('raft_optical_flow')

for entry in timeline:
    flow_vis = entry.output.metadata['visualisation']  # RGB image (H, W, 3)
    # Display or save visualisation
```

Flow visualisations use HSV colour wheel encoding:
- **Hue**: Flow direction (angle)
- **Saturation**: Full (255)
- **Value**: Flow magnitude

## Performance

Processing speed depends on:
- GPU availability (CUDA significantly faster)
- Model size (small vs large)
- Video resolution
- Number of RAFT iterations

Typical speeds (RAFT Small on RTX 3090):
- 1920×1080: ~20-30 FPS
- 1280×720: ~40-50 FPS
- 640×480: ~80-100 FPS

## Utility Functions

The `lucidity.flow_storage` module provides utilities for working with optical flow:

```python
from lucidity.flow_storage import (
    save_flow_flo,              # Save as .flo format
    load_flow_flo,              # Load .flo format
    save_flow_compressed,       # Save single frame compressed
    load_flow_compressed,       # Load single frame
    save_flow_sequence_compressed,  # Save sequence
    load_flow_sequence_compressed,  # Load sequence
)

# Save in Middlebury .flo format
save_flow_flo(flow, 'flow.flo')

# Load .flo file
flow = load_flow_flo('flow.flo')
```

## Integration with Other Models

Optical flow can be combined with other models:

```bash
python -m lucidity.cli process video.mp4 \
    --models raft_optical_flow,endomust_depth \
    --output-format compressed
```

The timeline synchronisation ensures all model outputs are aligned.

## Troubleshooting

### Out of Memory

If you encounter GPU memory errors:

1. Use smaller model: `"model_size": "small"`
2. Process fewer frames at once
3. Reduce `num_flow_updates`

### Slow Processing

1. Ensure CUDA is available: Check `torch.cuda.is_available()`
2. Use RAFT Small instead of Large
3. Reduce video resolution if possible

### Import Errors

Ensure torchvision is up to date:

```bash
pip install --upgrade torchvision>=0.12.0
```

## References

- RAFT Paper: [https://arxiv.org/abs/2003.12039](https://arxiv.org/abs/2003.12039)
- PyTorch RAFT: [https://pytorch.org/vision/stable/models/optical_flow.html](https://pytorch.org/vision/stable/models/optical_flow.html)
