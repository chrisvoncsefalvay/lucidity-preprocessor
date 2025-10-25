"""
Example sparse output model plugin.

This model demonstrates sparse outputs that only occur at certain events.
For example: scene detection, event detection, anomaly detection, etc.
"""

import numpy as np
from typing import Optional, Dict, Any

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType


class ExampleSparseModel(BaseModel):
    """
    Example model that produces sparse outputs.

    Outputs are only generated when certain conditions are met (e.g., scene changes).
    This is a dummy implementation that randomly triggers events.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        super().__init__(config)
        self.event_probability = config.get('event_probability', 0.05) if config else 0.05
        self.last_scene_frame = 0

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="example_sparse_model",
            version="0.1.0",
            description="Example sparse output model (detects scene changes)",
            author="Lucidity Team",
            output_type=OutputType.LABEL,
            output_frequency="event_based",  # Not regular frequency
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the model."""
        # In a real implementation, you might:
        # - Load a scene detection model
        # - Initialize feature extractors
        # - Set up similarity metrics
        self.last_scene_frame = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame for event detection.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput only when an event is detected, None otherwise
        """
        # Dummy implementation: randomly detect scene changes
        # In a real implementation, you would:
        # - Compare frame features with previous frames
        # - Detect significant changes
        # - Classify scene types

        is_scene_change = np.random.random() < self.event_probability

        if not is_scene_change:
            return None  # No output for this frame

        # Scene change detected - generate output
        scene_duration = frame_number - self.last_scene_frame
        self.last_scene_frame = frame_number

        # Calculate some dummy features
        mean_color = frame.mean(axis=(0, 1)).tolist()
        brightness = frame.mean()

        data = {
            "event_type": "scene_change",
            "scene_id": frame_number // 100,  # Dummy scene ID
            "scene_type": np.random.choice([
                "indoor", "outdoor", "transition", "black"
            ]),
            "confidence": float(np.random.uniform(0.7, 1.0)),
            "duration_frames": scene_duration,
            "features": {
                "mean_color_rgb": mean_color,
                "brightness": float(brightness),
            },
        }

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=data,
            confidence=data["confidence"],
            metadata={
                "frames_since_last_scene": scene_duration,
            },
        )

    def get_output_schema(self) -> Dict[str, Any]:
        """Return JSON schema describing the output format."""
        return {
            "type": "object",
            "description": "Scene change detection events",
            "properties": {
                "event_type": {"type": "string"},
                "scene_id": {"type": "integer"},
                "scene_type": {"type": "string"},
                "confidence": {"type": "number"},
                "duration_frames": {"type": "integer"},
                "features": {"type": "object"},
            },
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return ExampleSparseModel
