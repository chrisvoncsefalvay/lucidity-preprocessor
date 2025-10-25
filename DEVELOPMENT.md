# Development Guide

## Setup

### Installation for Development

```bash
# Clone or navigate to the repository
cd lucidity-preprocessor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check CLI is working
lucidity --help

# List available models
lucidity list-models --discover-dir ./examples
```

## Project Structure

```
lucidity-preprocessor/
├── lucidity/              # Core package
│   ├── __init__.py
│   ├── base_model.py     # Plugin interface
│   ├── video.py          # Video I/O and metadata
│   ├── timeline.py       # Time synchronization
│   ├── manifest.py       # Output manifests
│   ├── plugin_manager.py # Plugin discovery
│   ├── processor.py      # Main processing pipeline
│   └── cli.py            # Command-line interface
├── models/               # Local model plugins directory
├── examples/             # Example model implementations
├── setup.py
├── requirements.txt
└── README.md
```

## Core Components

### 1. Video Processing (`video.py`)

Handles video I/O with frame-accurate timing:

- **VideoMetadata**: Video file metadata (fps, resolution, duration)
- **VideoReader**: Iterate through frames with timestamps

### 2. Timeline System (`timeline.py`)

Synchronizes outputs from different models:

- **Timeline**: Store and align outputs at different frequencies
- **SynchronizedFrame**: Frame with all model outputs aligned
- Interpolation methods: nearest, forward_fill, none

### 3. Plugin System (`base_model.py`, `plugin_manager.py`)

Modular model plugin architecture:

- **BaseModel**: Abstract base class for all models
- **PluginManager**: Discovery and loading of plugins
- Entry point system for packaging

### 4. Manifest System (`manifest.py`)

Track all outputs and metadata:

- **ProcessingManifest**: Complete processing record
- **ManifestBuilder**: Construct manifests during processing
- JSON format for easy inspection

## Creating Models

### Model Interface

All models must implement:

```python
class MyModel(BaseModel):
    def get_metadata(self) -> ModelMetadata:
        """Return model information"""

    def initialize(self) -> None:
        """Load model weights, setup"""

    def process_frame(self, frame, timestamp, frame_number) -> Optional[ModelOutput]:
        """Process a single frame"""

    def cleanup(self) -> None:
        """Release resources"""
```

### Optional Methods

```python
def should_process_frame(self, timestamp, frame_number) -> bool:
    """Control which frames to process"""

def get_output_schema(self) -> Dict:
    """Define output data schema"""
```

## Testing

### Manual Testing

```bash
# Test with example models
lucidity process test_video.mp4 \\
  --models example_frame_model \\
  --discover-dir ./examples \\
  --output ./test_output

# Check the manifest
lucidity show-manifest ./test_output/manifest.json
```

### Development Workflow

1. Create model in `models/` or `examples/`
2. Implement required methods
3. Test with `--discover-dir`
4. Package as entry point for distribution

## Plugin Discovery

### Three Ways to Load Plugins

1. **Entry Points** (production):
   ```python
   # In setup.py of your model package
   entry_points={
       "lucidity.models": [
           "my_model = my_package:MyModel",
       ],
   }
   ```

2. **Directory Discovery** (development):
   ```bash
   lucidity process video.mp4 --discover-dir ./models
   ```

3. **Programmatic** (scripts):
   ```python
   from lucidity.plugin_manager import PluginManager
   manager = PluginManager()
   manager.register_plugin("my_model", MyModel)
   ```

## Time Synchronization

### Output Frequencies

Models can have different output frequencies:

- **Every frame**: `output_frequency="per_frame"`
- **Fixed rate**: `output_frequency="1.0"` (Hz), set `frame_rate`
- **Event-based**: `output_frequency="event_based"`

### Synchronization Methods

```python
# Get outputs for specific frame
sync_frame = timeline.get_outputs_for_frame(
    frame_number=100,
    interpolation="nearest"  # or "forward_fill", "none"
)

# Access model outputs
pose_output = sync_frame.get_output("pose_model")
scene_output = sync_frame.get_output("scene_model")
```

## Output Formats

### By Output Type

| Type | Saved As | Format |
|------|----------|--------|
| FRAME | Individual files or video | .npy, .png |
| KEYPOINTS | JSON | Structured coordinates |
| BBOX | JSON | Box coordinates |
| LABEL | JSON | Classification data |
| EMBEDDING | NumPy archive | .npz |
| CUSTOM | JSON | Any structure |

### Manifest Structure

```json
{
  "version": "1.0",
  "created_at": "2025-10-25T...",
  "video_metadata": {...},
  "models": {
    "model_name": {
      "model_metadata": {...},
      "output_files": [...],
      "total_outputs": 1000,
      "first_timestamp": 0.0,
      "last_timestamp": 10.0
    }
  }
}
```

## Best Practices

### Performance

1. Use `should_process_frame()` to skip unnecessary processing
2. Process frames incrementally (don't load entire video)
3. Use appropriate output frequencies
4. Clean up resources in `cleanup()`

### Memory

1. Don't accumulate all outputs in memory
2. Save outputs incrementally if needed
3. Use generators where possible
4. Be mindful of model memory usage

### Accuracy

1. Always use provided `timestamp` and `frame_number`
2. Don't calculate your own timing
3. Trust the timeline synchronization
4. Include confidence scores

### Modularity

1. Models should be independent
2. Don't depend on other model outputs
3. Use configuration for model parameters
4. Document output schema

## Common Patterns

### Sparse Outputs

```python
def process_frame(self, frame, timestamp, frame_number):
    if not self.is_interesting_frame(frame):
        return None  # No output

    return ModelOutput(...)
```

### Frame Sampling

```python
def should_process_frame(self, timestamp, frame_number):
    # Process every 5th frame
    return frame_number % 5 == 0
```

### Stateful Processing

```python
def __init__(self, config):
    super().__init__(config)
    self.previous_frame = None

def process_frame(self, frame, timestamp, frame_number):
    if self.previous_frame is not None:
        # Compare with previous frame
        diff = self.compute_diff(frame, self.previous_frame)

    self.previous_frame = frame
    return ModelOutput(...)
```

## Troubleshooting

### Plugin Not Found

1. Check plugin name matches `get_metadata().name`
2. Use `--discover-dir` to specify directory
3. Verify entry point is correct
4. Check for import errors in plugin

### Timing Issues

1. Always use provided `timestamp` and `frame_number`
2. Don't calculate your own timing
3. Check video fps in metadata
4. Use timeline synchronization methods

### Memory Issues

1. Process frames one at a time
2. Clear intermediate buffers
3. Implement `cleanup()` properly
4. Check model memory usage

## Next Steps

1. Create your first model using examples as templates
2. Test with small videos first
3. Add proper configuration support
4. Package as entry point for distribution
5. Add tests and documentation
