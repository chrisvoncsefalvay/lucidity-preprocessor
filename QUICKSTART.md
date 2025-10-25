# Quick Start Guide

Get started with Lucidity Preprocessor in 5 minutes.

## Installation

```bash
# Install the package
pip install -e .
```

## Basic Usage

### 1. List Available Models

```bash
# See example models
lucidity list-models --discover-dir ./examples

# Get detailed information
lucidity list-models --discover-dir ./examples --verbose
```

### 2. Process a Video

```bash
# Process with a single model
lucidity process your_video.mp4 \
  --models example_frame_model \
  --discover-dir ./examples \
  --output ./output

# Process with multiple models
lucidity process your_video.mp4 \
  --models example_frame_model,example_keypoint_model,example_sparse_model \
  --discover-dir ./examples \
  --output ./output
```

### 3. View Results

```bash
# View the processing manifest
lucidity show-manifest ./output/manifest.json

# Outputs are organized by model
ls ./output/
#   manifest.json
#   example_frame_model/
#   example_keypoint_model/
#   example_sparse_model/
```

## Output Structure

After processing, you'll have:

```
output/
├── manifest.json                    # Complete processing record
├── example_frame_model/
│   ├── frames/                      # Frame outputs
│   │   ├── frame_000000.npy
│   │   ├── frame_000001.npy
│   │   └── ...
│   └── model_metadata.json
├── example_keypoint_model/
│   ├── outputs.json                 # Keypoint detections
│   └── model_metadata.json
└── example_sparse_model/
    ├── outputs.json                 # Scene change events
    └── model_metadata.json
```

## Creating Your First Model

### Option 1: Use the Template Generator

```bash
lucidity create-template my_pose_detector \
  --type keypoints \
  --frequency per_frame
```

This creates `my_pose_detector_model.py` with TODOs to fill in.

### Option 2: Copy an Example

```bash
cp examples/example_keypoint_model.py models/my_model.py
# Edit models/my_model.py and implement your model
```

### Test Your Model

```bash
lucidity process test_video.mp4 \
  --models my_model \
  --discover-dir ./models \
  --output ./test_output
```

## Model Configuration

Create a `config.json` file:

```json
{
  "example_keypoint_model": {
    "num_keypoints": 21,
    "process_every_n_frames": 2
  },
  "my_model": {
    "threshold": 0.8,
    "batch_size": 1
  }
}
```

Use it:

```bash
lucidity process video.mp4 \
  --models example_keypoint_model,my_model \
  --model-config config.json \
  --discover-dir ./examples \
  --discover-dir ./models
```

## Working with Outputs

### Load the Manifest

```python
from lucidity.manifest import load_manifest

manifest = load_manifest("./output/manifest.json")

# Access video metadata
print(f"Duration: {manifest.video_metadata.duration}s")
print(f"FPS: {manifest.video_metadata.fps}")

# Access model outputs
for model_name, model_info in manifest.models.items():
    print(f"{model_name}: {model_info.total_outputs} outputs")
```

### Load Model Outputs

```python
import json
import numpy as np

# Load keypoint data
with open("./output/example_keypoint_model/outputs.json") as f:
    keypoints = json.load(f)

for item in keypoints:
    print(f"Frame {item['frame_number']}: {len(item['data']['keypoints'])} keypoints")

# Load frame data
frame = np.load("./output/example_frame_model/frames/frame_000000.npy")
print(f"Frame shape: {frame.shape}")
```

### Use Timeline Synchronization

```python
from lucidity.video import VideoReader
from lucidity.timeline import Timeline
from lucidity.plugin_manager import PluginManager
from lucidity.processor import VideoProcessor

# If you need to align outputs programmatically
reader = VideoReader("video.mp4")
timeline = Timeline(reader.metadata.fps, reader.metadata.total_frames)

# Add outputs from your processing...
# Then get synchronized frame
sync_frame = timeline.get_outputs_for_frame(100, interpolation="nearest")

# Access outputs from different models
pose = sync_frame.get_output("pose_model")
scene = sync_frame.get_output("scene_model")
```

## Common Use Cases

### 1. Pose Estimation on Every Frame

```python
# Create a model that inherits from BaseModel
# Set output_type=OutputType.KEYPOINTS
# Set output_frequency="per_frame"
# Return keypoints in process_frame()
```

### 2. Scene Detection (Sparse Events)

```python
# Set output_type=OutputType.LABEL
# Set output_frequency="event_based"
# Return None when no scene change
# Return output only on scene boundaries
```

### 3. Feature Extraction Every Second

```python
# Set output_type=OutputType.EMBEDDING
# Set frame_rate=1.0 (1 Hz)
# Implement should_process_frame() to sample correctly
```

### 4. Object Detection with Bounding Boxes

```python
# Set output_type=OutputType.BBOX
# Return bounding boxes in standard format
# Include confidence scores
```

## Tips

1. **Start Simple**: Test with short videos first (5-10 seconds)
2. **Use Examples**: The example models demonstrate all output types
3. **Check Manifest**: Always inspect the manifest.json to verify outputs
4. **Frame Timing**: Always use the provided timestamp and frame_number
5. **Cleanup**: Implement cleanup() to release model resources

## Next Steps

- Read [DEVELOPMENT.md](DEVELOPMENT.md) for detailed documentation
- Check [examples/README.md](examples/README.md) for model patterns
- See [README.md](README.md) for architecture overview

## Getting Help

- Check the examples in `./examples/`
- Read the model interface in `lucidity/base_model.py`
- Use `--help` on any command:
  ```bash
  lucidity --help
  lucidity process --help
  ```
