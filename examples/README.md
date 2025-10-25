# Example Model Plugins

This directory contains example model plugins demonstrating different output types and processing patterns.

## Available Examples

### 1. Frame-based Model (`example_frame_model.py`)

Demonstrates processing every frame and outputting frame-based data.

**Use cases:**
- Segmentation masks
- Style transfer
- Super-resolution
- Color grading
- Frame enhancement

**Output:** Frame data (images/arrays)

### 2. Keypoint Detection Model (`example_keypoint_model.py`)

Demonstrates detecting keypoints in frames.

**Use cases:**
- Pose estimation
- Hand tracking
- Facial landmarks
- Object keypoints

**Output:** Structured keypoint data with coordinates and confidence scores

**Features:**
- Configurable number of keypoints
- Frame sampling (process every N frames)
- Skeleton connections for visualization

### 3. Sparse Output Model (`example_sparse_model.py`)

Demonstrates event-based sparse outputs.

**Use cases:**
- Scene detection
- Event detection
- Anomaly detection
- Shot boundary detection

**Output:** Labels/events only when conditions are met

**Features:**
- Event-based triggering
- Variable output frequency
- Rich metadata per event

## Using the Examples

### 1. Test with an example model

```bash
# Process a video with the example frame model
lucidity process video.mp4 \\
  --models example_frame_model \\
  --discover-dir ./examples \\
  --output ./test_output

# Process with multiple example models
lucidity process video.mp4 \\
  --models example_frame_model,example_keypoint_model,example_sparse_model \\
  --discover-dir ./examples \\
  --output ./test_output
```

### 2. List available models

```bash
lucidity list-models --discover-dir ./examples --verbose
```

### 3. Get model information

```bash
lucidity model-info example_keypoint_model --discover-dir ./examples
```

## Creating Your Own Model

### Option 1: Use the template generator

```bash
lucidity create-template my_model --type keypoints --frequency per_frame
```

### Option 2: Copy and modify an example

1. Copy the example that best matches your use case
2. Rename the class and file
3. Implement the TODOs:
   - `get_metadata()`: Update model information
   - `initialize()`: Load your model weights
   - `process_frame()`: Run your inference
   - `cleanup()`: Release resources

### Model Configuration

Models can accept configuration via JSON:

```json
{
  "example_keypoint_model": {
    "num_keypoints": 21,
    "process_every_n_frames": 2
  },
  "example_sparse_model": {
    "event_probability": 0.1
  }
}
```

Use with:

```bash
lucidity process video.mp4 \\
  --models example_keypoint_model \\
  --model-config config.json \\
  --discover-dir ./examples
```

## Output Types

| Type | Description | Example Use Cases |
|------|-------------|-------------------|
| FRAME | Image/frame data | Segmentation, style transfer |
| KEYPOINTS | Structured coordinates | Pose, landmarks, tracking |
| BBOX | Bounding boxes | Object detection |
| LABEL | Classifications | Scene detection, events |
| EMBEDDING | Feature vectors | Similarity, retrieval |
| TIMESERIES | Sequential data | Audio features, motion |
| CUSTOM | Any structured data | Complex outputs |

## Output Frequencies

- `per_frame`: Process and output every frame
- `per_second`: Fixed frequency (specify `frame_rate` in metadata)
- `event_based`: Sparse outputs based on conditions
- Custom: Implement `should_process_frame()` for custom logic

## Best Practices

1. **Efficiency**: Use `should_process_frame()` to skip unnecessary processing
2. **Memory**: Process frames incrementally, don't load entire video into memory
3. **Metadata**: Include rich metadata for downstream analysis
4. **Confidence**: Always include confidence scores when applicable
5. **Cleanup**: Release resources (models, buffers) in `cleanup()`
6. **Testing**: Test with videos of different resolutions and frame rates

## Installing Models as Packages

For production use, create a proper Python package:

```python
# setup.py
setup(
    name="lucidity-model-mymodel",
    entry_points={
        "lucidity.models": [
            "my_model = my_package.model:MyModel",
        ],
    },
)
```

Then install:

```bash
pip install lucidity-model-mymodel
```

Models will be automatically discovered without needing `--discover-dir`.
