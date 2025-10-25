"""
Example frame-based model plugin.

This model demonstrates processing every frame and outputting frame-based data.
For example: segmentation masks, style transfer, super-resolution, etc.
"""

import numpy as np
from typing import Optional

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType


class ExampleFrameModel(BaseModel):
    """
    Example model that processes every frame and outputs frame data.

    This is a dummy implementation that just applies a simple filter.
    Replace with your actual model implementation.
    """

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="example_frame_model",
            version="0.1.0",
            description="Example frame-based processing model (applies grayscale conversion)",
            author="Lucidity Team",
            output_type=OutputType.FRAME,
            output_frequency="per_frame",
            frame_rate=None,  # Same as input video
            dependencies=[],
        )

    def initialize(self) -> None:
        """Initialize the model."""
        # No initialization needed for this simple example
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput with processed frame
        """
        # Simple example: convert to grayscale
        grayscale = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])

        # Return as single-channel image
        output_frame = grayscale.astype(np.uint8)

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=output_frame,
            confidence=1.0,
            metadata={
                "output_shape": output_frame.shape,
                "processing": "grayscale_conversion",
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return ExampleFrameModel
