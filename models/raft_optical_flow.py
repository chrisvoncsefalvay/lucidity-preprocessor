"""
RAFT optical flow model plugin for lucidity.

This model uses the RAFT (Recurrent All-Pairs Field Transforms) model from
torchvision to compute sparse optical flow within the circular mask region.

The model:
- Computes dense optical flow between consecutive frames
- Downsamples to a sparse grid for efficiency
- Filters vectors to only those inside the circular mask
- Stores flow vectors with their coordinates

Reference: https://pytorch.org/vision/stable/models/optical_flow.html
"""

import numpy as np
import torch
import cv2
from typing import Optional, Tuple
import torchvision.transforms.functional as F

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType
from lucidity.masking import CircularMask

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
    RAFT optical flow model for sparse motion estimation within mask.

    This model computes optical flow between consecutive frames using the
    RAFT architecture, then downsamples and filters to only vectors inside
    the circular mask region. The first frame produces no output.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model = None
        self.device = None
        self.previous_frame = None
        self.mask = None
        self.frame_for_mask = None

        # Model configuration
        self.model_size = self.config.get('model_size', 'small')  # 'small' or 'large'
        self.weights = self.config.get('weights', 'default')  # 'default' or path to custom weights

        # Flow computation parameters
        self.num_flow_updates = self.config.get('num_flow_updates', 12)  # RAFT iterations

        # Sparse flow configuration
        self.stride = self.config.get('stride', 16)  # Sample every Nth pixel (8, 16, or 32)
        self.mask_threshold = self.config.get('mask_threshold', 30)  # Black threshold for masking
        self.mask_method = self.config.get('mask_method', 'hough')  # 'hough' or 'contour'

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="raft_optical_flow",
            version="2.0.0",
            description=f"RAFT sparse optical flow (stride={self.stride}) within circular mask",
            author="Princeton Vision Lab (RAFT), adapted for Lucidity",
            output_type=OutputType.CUSTOM,  # Sparse vectors, not full frame
            output_frequency="per_frame",
            frame_rate=None,  # Same as input video
            dependencies=["torch", "torchvision>=0.12.0", "opencv-python", "numpy"],
        )

    def initialize(self) -> None:
        """Initialise the RAFT model and load pretrained weights."""
        print(f"Initialising RAFT sparse optical flow model...")
        print(f"Configuration:")
        print(f"  - Model size: {self.model_size}")
        print(f"  - Flow updates: {self.num_flow_updates}")
        print(f"  - Sampling stride: {self.stride} pixels")
        print(f"  - Mask threshold: {self.mask_threshold}")
        print(f"  - Mask method: {self.mask_method}")

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
        Process a single frame and generate sparse optical flow within mask.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput with sparse flow vectors, or None for the first frame
        """
        if self.model is None:
            raise RuntimeError("Model not initialised. Call initialize() first.")

        # Detect mask on first frame
        if self.mask is None:
            self._detect_mask(frame)

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
        flow_dense = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Downsample and filter flow to sparse vectors inside mask
        sparse_flow = self._extract_sparse_flow(flow_dense)

        # Update previous frame for next iteration
        self.previous_frame = current_tensor

        # Compute statistics
        if len(sparse_flow['x']) > 0:
            magnitudes = np.sqrt(sparse_flow['u']**2 + sparse_flow['v']**2)
            mean_mag = float(magnitudes.mean())
            max_mag = float(magnitudes.max())
            std_mag = float(magnitudes.std())
        else:
            mean_mag = max_mag = std_mag = 0.0

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=sparse_flow,  # Sparse flow vectors with coordinates
            confidence=None,
            metadata={
                "num_vectors": len(sparse_flow['x']),
                "stride": self.stride,
                "mask_radius": float(self.mask.radius) if self.mask else 0.0,
                "mask_centre": [float(self.mask.centre[0]), float(self.mask.centre[1])] if self.mask else [0.0, 0.0],
                "flow_magnitude_mean": mean_mag,
                "flow_magnitude_max": max_mag,
                "flow_magnitude_std": std_mag,
            },
        )

    def _detect_mask(self, frame: np.ndarray) -> None:
        """
        Detect circular mask from the first frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
        """
        from lucidity.masking import EndoscopicMaskDetector

        print(f"Detecting circular mask for optical flow filtering...")

        detector = EndoscopicMaskDetector(
            n_frames=1,  # Just use this one frame
            black_threshold=self.mask_threshold,
            circle_fit_method=self.mask_method,
        )

        # Use single frame for detection (needs to be array with shape (1, H, W, 3))
        frames_array = np.expand_dims(frame, axis=0)
        self.mask = detector.detect_mask_from_frames(frames_array)

        if self.mask:
            print(f"  - Mask detected: centre=({self.mask.centre[0]:.1f}, {self.mask.centre[1]:.1f}), radius={self.mask.radius:.1f}")
        else:
            print(f"  - No mask detected, using full frame")

    def _extract_sparse_flow(self, flow_dense: np.ndarray) -> dict:
        """
        Extract sparse flow vectors from dense flow field.

        Downsamples the flow field by stride and filters to only include
        vectors inside the circular mask.

        Args:
            flow_dense: Dense flow field (H, W, 2)

        Returns:
            Dictionary with 'x', 'y', 'u', 'v' arrays for sparse vectors
        """
        h, w = flow_dense.shape[:2]

        # Create sampling grid
        y_coords = np.arange(0, h, self.stride)
        x_coords = np.arange(0, w, self.stride)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Flatten coordinates
        x_flat = xx.flatten()
        y_flat = yy.flatten()

        # Sample flow at these coordinates
        u_flat = flow_dense[y_flat, x_flat, 0]
        v_flat = flow_dense[y_flat, x_flat, 1]

        # Filter to only include vectors inside mask
        if self.mask is not None:
            # Calculate distance from mask centre
            dx = x_flat - self.mask.centre[0]
            dy = y_flat - self.mask.centre[1]
            dist = np.sqrt(dx**2 + dy**2)

            # Keep only vectors inside mask
            inside_mask = dist <= self.mask.radius

            x_flat = x_flat[inside_mask]
            y_flat = y_flat[inside_mask]
            u_flat = u_flat[inside_mask]
            v_flat = v_flat[inside_mask]

        return {
            'x': x_flat.astype(np.float32),
            'y': y_flat.astype(np.float32),
            'u': u_flat.astype(np.float32),
            'v': v_flat.astype(np.float32),
        }

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
