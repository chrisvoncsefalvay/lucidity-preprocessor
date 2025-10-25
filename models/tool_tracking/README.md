# ConvLSTM Surgical Tool Tracker

This model detects and tracks surgical instruments in laparoscopic videos using a weakly supervised deep learning approach combining ResNet and ConvLSTM.

## Overview

The ConvLSTM tool tracker identifies and localises surgical tools in endoscopic video frames. It uses:
- **ResNet-18** for feature extraction
- **ConvLSTM** for temporal modelling and tracking
- **Fully Convolutional Network** for localisation

## Model Details

- **Input**: RGB video frames (automatically resized to 480×854)
- **Output**: Bounding boxes and confidence scores for detected surgical tools
- **Tool Classes**: 7 surgical instruments
  1. Grasper
  2. Bipolar forceps
  3. Hook
  4. Scissors
  5. Clipper
  6. Irrigator
  7. Specimen bag

## Usage

### Basic Usage

```bash
lucidity process video.mp4 --models tool_tracking
```

### With Configuration

```bash
lucidity process video.mp4 --models tool_tracking --model-config config.json
```

Example `config.json`:
```json
{
  "tool_tracking": {
    "confidence_threshold": 0.5,
    "checkpoint_path": "models/tool_tracking/checkpoints/ckpt/model.ckpt"
  }
}
```

### With Frame Range

```bash
lucidity process video.mp4 --models tool_tracking --start-frame 100 --end-frame 500
```

## Output Format

The model produces JSON output with the following structure:

```json
{
  "timestamp": 1.5,
  "frame_number": 45,
  "data": {
    "detections": [
      {
        "tool": "grasper",
        "confidence": 0.85,
        "bbox": [120, 230, 180, 290],
        "centre": [150, 260],
        "heatmap": [[...]]
      }
    ],
    "num_tools": 1
  },
  "confidence": 0.85,
  "metadata": {
    "seek": 45,
    "all_probabilities": {
      "grasper": 0.85,
      "bipolar": 0.12,
      "hook": 0.05,
      ...
    }
  }
}
```

## Installation

The model requires TensorFlow 1.x. Install dependencies:

```bash
pip install tensorflow==1.15.0
```

Note: TensorFlow 1.15 is the last 1.x release and may require Python 3.7 or earlier.

## Pretrained Weights

The pretrained weights are located in:
```
models/tool_tracking/checkpoints/ckpt/model.ckpt
```

These weights were trained on the Cholec80 dataset.

## Citation

If you use this model, please cite:

```
@article{nwoye2019weakly,
  title={Weakly supervised convolutional LSTM approach for tool tracking in laparoscopic videos},
  author={Nwoye, Chinedu Innocent and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  volume={14},
  number={6},
  pages={1059--1067},
  year={2019},
  publisher={Springer}
}
```

## License

This code is based on the original implementation by CAMMA-public and is intended for non-commercial scientific research use only (CC BY-NC-SA 4.0).

## Repository

Original implementation: https://github.com/CAMMA-public/ConvLSTM-Surgical-Tool-Tracker
