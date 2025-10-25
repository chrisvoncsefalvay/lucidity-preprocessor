"""
Example model with endoscopic masking.

This model demonstrates how to detect and apply circular masks to endoscopic videos,
removing the black border around the circular endoscopic image.
"""

import numpy as np
from typing import Optional
import cv2

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType
from lucidity.masking import CircularMask, EndoscopicMaskDetector


class ExampleMaskedModel(BaseModel):
    """
    Example model that automatically detects and applies endoscopic masks.

    This model:
    1. Analyses the first N frames to detect the circular endoscopic region
    2. Creates a clean circular mask
    3. Applies the mask to all processed frames
    4. Outputs masked frames suitable for downstream ML inference
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise the masked model.

        Config options:
            - mask_n_frames: Number of frames to analyse for mask detection (default: 10)
            - mask_black_threshold: Threshold for black pixels (default: 30)
            - mask_min_valid_ratio: Min ratio of non-black frames (default: 0.3)
            - mask_morph_kernel_size: Kernel size for morphology (default: 5)
            - mask_circle_fit_method: Circle fitting method (default: 'hough')
        """
        super().__init__(config)
        self.mask: Optional[CircularMask] = None
        self.frames_for_mask = []
        self.mask_n_frames = self.config.get('mask_n_frames', 10)

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="example_masked_model",
            version="0.1.0",
            description="Example model with automatic endoscopic masking",
            author="Lucidity Team",
            output_type=OutputType.FRAME,
            output_frequency="per_frame",
            frame_rate=None,
            dependencies=["opencv-python", "numpy"],
        )

    def initialize(self) -> None:
        """Initialise the model."""
        self.frames_for_mask = []
        self.mask = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame with masking.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput with masked frame
        """
        # Collect initial frames for mask detection
        if self.mask is None:
            if len(self.frames_for_mask) < self.mask_n_frames:
                self.frames_for_mask.append(frame.copy())

            # Once we have enough frames, detect the mask
            if len(self.frames_for_mask) == self.mask_n_frames:
                self._detect_mask()

            # Don't process frames until we have the mask
            if self.mask is None:
                return None

        # Apply the mask to the frame
        masked_frame = self.mask.apply(frame)

        # Example processing: enhance contrast in the masked region
        # (Replace this with your actual ML model inference)
        processed_frame = self._enhance_contrast(masked_frame)

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=processed_frame,
            confidence=1.0,
            metadata={
                "mask_centre": self.mask.centre,
                "mask_radius": self.mask.radius,
                "output_shape": processed_frame.shape,
            },
        )

    def _detect_mask(self) -> None:
        """Detect the circular mask from collected frames."""
        if not self.frames_for_mask:
            return

        frames_array = np.array(self.frames_for_mask)

        # Create detector with config parameters
        detector = EndoscopicMaskDetector(
            n_frames=len(self.frames_for_mask),
            black_threshold=self.config.get('mask_black_threshold', 30),
            min_valid_ratio=self.config.get('mask_min_valid_ratio', 0.3),
            morph_kernel_size=self.config.get('mask_morph_kernel_size', 5),
            circle_fit_method=self.config.get('mask_circle_fit_method', 'hough'),
        )

        self.mask = detector.detect_mask_from_frames(frames_array)

        # Clear frames from memory
        self.frames_for_mask = []

        print(f"Mask detected: centre={self.mask.centre}, radius={self.mask.radius:.1f}")

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Example processing: enhance contrast using CLAHE.

        Replace this with your actual model inference.

        Args:
            frame: Masked RGB frame

        Returns:
            Processed frame
        """
        # Convert to LAB colour space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.frames_for_mask = []
        self.mask = None


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return ExampleMaskedModel
