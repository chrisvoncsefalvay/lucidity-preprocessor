"""
RAFT optical flow model plugin for lucidity.

This model uses the RAFT (Recurrent All-Pairs Field Transforms) model from
torchvision to compute dense optical flow between consecutive frames.

Reference: https://pytorch.org/vision/stable/models/optical_flow.html
"""

import numpy as np
import torch
import cv2
from typing import Optional
import torchvision.transforms.functional as F

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType

try:
    from torchvision.models.optical_flow import raft_small, raft_large
    from torchvision.models.optical_flow import Raft_Small_Weights, Raft_Large_Weights
except ImportError as e:
    print(f"Error importing RAFT from torchvision: {e}")
    print("Please install torchvision>=0.12.0: pip install torchvision>=0.12.0")
    raise


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB visualisation using HSV colour wheel.

    Args:
        flow: Optical flow (H, W, 2) with (u, v) components

    Returns:
        RGB image (H, W, 3) visualising the flow
    """
    h, w = flow.shape[:2]

    # Calculate flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


class RAFTOpticalFlowModel(BaseModel):
    """
    RAFT optical flow model for dense motion estimation.

    This model computes optical flow between consecutive frames using the
    RAFT architecture. The first frame produces no output (no previous frame).
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model = None
        self.device = None
        self.previous_frame = None

        # Model configuration
        self.model_size = self.config.get('model_size', 'small')  # 'small' or 'large'
        self.weights = self.config.get('weights', 'default')  # 'default' or path to custom weights

        # Flow computation parameters
        self.num_flow_updates = self.config.get('num_flow_updates', 12)  # RAFT iterations

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="raft_optical_flow",
            version="1.0.0",
            description="RAFT dense optical flow estimation between consecutive frames",
            author="Princeton Vision Lab (RAFT), adapted for Lucidity",
            output_type=OutputType.FRAME,
            output_frequency="per_frame",
            frame_rate=None,  # Same as input video
            dependencies=["torch", "torchvision>=0.12.0", "opencv-python", "numpy"],
        )

    def initialize(self) -> None:
        """Initialise the RAFT model and load pretrained weights."""
        print(f"Initialising RAFT optical flow model...")
        print(f"Configuration:")
        print(f"  - Model size: {self.model_size}")
        print(f"  - Flow updates: {self.num_flow_updates}")

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  - Device: {self.device}")

        # Load model
        if self.model_size == 'small':
            if self.weights == 'default':
                weights = Raft_Small_Weights.DEFAULT
                print(f"  - Loading pretrained RAFT Small weights")
            else:
                weights = None
                print(f"  - Loading custom weights from: {self.weights}")
            self.model = raft_small(weights=weights)
        elif self.model_size == 'large':
            if self.weights == 'default':
                weights = Raft_Large_Weights.DEFAULT
                print(f"  - Loading pretrained RAFT Large weights")
            else:
                weights = None
                print(f"  - Loading custom weights from: {self.weights}")
            self.model = raft_large(weights=weights)
        else:
            raise ValueError(f"Invalid model_size: {self.model_size}. Must be 'small' or 'large'")

        # Load custom weights if specified
        if self.weights != 'default':
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device))

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Reset previous frame
        self.previous_frame = None

        print("RAFT model initialised successfully")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame and generate optical flow from previous frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput with optical flow data, or None for the first frame
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call initialize() first.")

        # Convert frame to tensor
        current_tensor = self._preprocess_frame(frame)

        # Skip first frame (no previous frame to compare)
        if self.previous_frame is None:
            self.previous_frame = current_tensor
            return None

        # Compute optical flow from previous to current frame
        with torch.no_grad():
            # RAFT expects frames in range [0, 255]
            prev_frame_255 = self.previous_frame * 255.0
            curr_frame_255 = current_tensor * 255.0

            # Compute flow
            flow_predictions = self.model(prev_frame_255, curr_frame_255)

            # Get the final flow prediction (after all iterations)
            flow = flow_predictions[-1]  # Shape: (1, 2, H, W)

        # Convert to numpy (1, 2, H, W) -> (H, W, 2)
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Compute flow statistics
        flow_magnitude = np.sqrt(flow_np[..., 0]**2 + flow_np[..., 1]**2)

        # Generate visualisation
        flow_vis = flow_to_rgb(flow_np)

        # Update previous frame for next iteration
        self.previous_frame = current_tensor

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=flow_np,  # Return the actual flow field (H, W, 2)
            confidence=None,  # RAFT doesn't provide per-pixel confidence
            metadata={
                "flow_shape": flow_np.shape,
                "flow_magnitude_mean": float(flow_magnitude.mean()),
                "flow_magnitude_max": float(flow_magnitude.max()),
                "flow_magnitude_std": float(flow_magnitude.std()),
                "flow_u_range": [float(flow_np[..., 0].min()), float(flow_np[..., 0].max())],
                "flow_v_range": [float(flow_np[..., 1].min()), float(flow_np[..., 1].max())],
                "visualisation": flow_vis,  # RGB visualisation
            },
        )

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for RAFT model.

        Args:
            frame: RGB frame as numpy array (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, H, W) normalised to [0, 1]
        """
        # Convert to float and normalise to [0, 1]
        frame_float = frame.astype(np.float32) / 255.0

        # Convert to tensor: (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.previous_frame is not None:
            del self.previous_frame
            self.previous_frame = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("RAFT model cleaned up")


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return RAFTOpticalFlowModel
