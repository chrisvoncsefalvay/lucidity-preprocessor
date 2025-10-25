# Lucidity Preprocessor

A modular video ML inference pipeline with self-discovering model plugins.

## Features

- Process MP4 videos with multiple ML models
- Modular plugin system for easy model integration
- Automatic time synchronization across different output types
- Frame-accurate alignment for seamless overlay/switching
- JSON manifest for complete output tracking
- Self-discovery of model plugins as Python packages

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

# List available models
lucidity list-models

# Get model info
lucidity model-info pose_detection
```

## Creating Model Plugins

See `examples/example_model.py` for a template. Model plugins should:
1. Inherit from `BaseModel`
2. Implement required methods
3. Be installable as a Python package with entry point
4. Follow the output format specification
