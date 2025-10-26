# Lucidity Preprocessor

A modular video ML inference pipeline with self-discovering model plugins.

## Features

- Process MP4 videos with multiple ML models
- Modular plugin system for easy model integration
- Automatic time synchronization across different output types
- Frame-accurate alignment for seamless overlay/switching
- JSON manifest for complete output tracking
- Self-discovery of model plugins as Python packages
- Endoscopic video masking for circular region detection
- Sparse optical flow for efficient motion analysis

## Architecture

### Core Components

- **Video Processing**: Extract metadata, frames, and timing information
- **Timeline System**: Synchronize outputs across different frequencies
- **Plugin Manager**: Auto-discover and load model plugins
- **Manifest Generator**: Create comprehensive JSON output manifests

### Plugin System

Models are implemented as plugins following a standard interface. Each plugin can:
- Output different data types (frames, keypoints, embeddings, etc.)
- Run at different frequencies (every frame, every N seconds, etc.)
- Specify dependencies and requirements
- Include metadata about outputs

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Process video with specific models
lucidity process video.mp4 --models pose_detection,scene_segmentation --output ./results

# Process with endoscopic masking
lucidity process video.mp4 --models my_model --mask --output ./results

# Process with custom masking parameters
lucidity process video.mp4 --models my_model --mask --mask-frames 20 --mask-threshold 40

# List available models
lucidity list-models

# Get model info
lucidity model-info pose_detection
```

### Masking options

When processing endoscopic videos with the `--mask` flag, the following options are available:

- `--mask`: Enable automatic circular region detection and masking
- `--mask-frames N`: Number of initial frames to analyse for mask detection (default: 10)
- `--mask-threshold N`: Pixel intensity threshold for black regions (0-255, default: 30)
- `--mask-method METHOD`: Circle fitting method, either `hough` or `contour` (default: hough)

## Creating model plugins

See `examples/example_model.py` for a template. Model plugins should:
1. Inherit from `BaseModel`
2. Implement required methods
3. Be installable as a Python package with entry point
4. Follow the output format specification

## Endoscopic masking

Lucidity includes automatic masking for endoscopic videos that display circular images on black backgrounds. The masking system:

- Automatically detects the circular endoscopic region from the first N frames
- Applies morphological operations to create clean, non-pixelated borders
- Provides a simple API for applying masks to frames during inference

For detailed documentation, see [MASKING.md](MASKING.md).

Quick example:

```python
from lucidity.masking import detect_mask_from_video

# Detect mask from video
mask = detect_mask_from_video("endoscopic_video.mp4", n_frames=10)

# Apply to frames
masked_frame = mask.apply(frame)
```

See [examples/example_masked_model.py](examples/example_masked_model.py) for a complete model implementation.

## Sparse optical flow

Lucidity includes sparse optical flow processing using RAFT for efficient motion analysis in surgical/endoscopic video. The optical flow system:

- Computes dense optical flow using RAFT (Recurrent All-Pairs Field Transforms)
- Downsamples to sparse grid with configurable stride (8, 16, or 32 pixels)
- Filters to only vectors inside the circular endoscopic mask
- Achieves 300-800x storage reduction vs dense flow
- Provides visualisation utilities for quiver plots and colour-coded flow

For detailed documentation, see [OPTICAL_FLOW.md](OPTICAL_FLOW.md).

Quick CLI example:

```bash
# Process with sparse optical flow
lucidity process video.mp4 --models raft_optical_flow --output ./results

# Custom stride for denser sampling
lucidity process video.mp4 --models raft_optical_flow --flow-stride 8

# Combine with depth estimation
lucidity process video.mp4 --models raft_optical_flow,endomust_depth
```

See [examples/optical_flow_cli_example.sh](examples/optical_flow_cli_example.sh) for more examples.
