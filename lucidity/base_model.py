"""Base model interface for Lucidity plugins."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel, Field
import numpy as np


class OutputType(str, Enum):
    """Types of outputs a model can produce."""
    FRAME = "frame"  # Image/frame output (e.g., segmentation masks)
    KEYPOINTS = "keypoints"  # Structured keypoint data
    EMBEDDING = "embedding"  # Feature vectors
    BBOX = "bbox"  # Bounding boxes
    LABEL = "label"  # Classification labels
    TIMESERIES = "timeseries"  # Time-series data
    CUSTOM = "custom"  # Custom structured data


class ModelMetadata(PydanticBaseModel):
    """Metadata about a model plugin."""
    name: str = Field(..., description="Unique model identifier")
    version: str = Field(..., description="Model version")
    description: str = Field(..., description="Human-readable description")
    author: Optional[str] = Field(None, description="Model author")
    output_type: OutputType = Field(..., description="Type of output produced")
    output_frequency: str = Field(..., description="Output frequency (e.g., 'per_frame', 'per_second', 'per_scene')")
    frame_rate: Optional[float] = Field(None, description="Expected output frame rate (Hz), if fixed")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")

    class Config:
        use_enum_values = True


class ModelOutput(PydanticBaseModel):
    """Standardized output format from a model."""
    timestamp: float = Field(..., description="Timestamp in seconds from video start")
    frame_number: Optional[int] = Field(None, description="Corresponding frame number in source video")
    data: Union[np.ndarray, Dict[str, Any], List[Any]] = Field(..., description="The actual output data")
    confidence: Optional[float] = Field(None, description="Confidence score if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True


class BaseModel(ABC):
    """
    Abstract base class for all Lucidity model plugins.

    All model plugins must inherit from this class and implement the required methods.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.

        Args:
            config: Optional configuration dictionary for the model
        """
        self.config = config or {}
        self._metadata = None

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Return metadata about this model.

        Returns:
            ModelMetadata object describing the model
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model (load weights, setup, etc.).

        This is called once before processing begins.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp: float, frame_number: int) -> Optional[ModelOutput]:
        """
        Process a single frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput if the model produces output for this frame, None otherwise
        """
        pass

    def should_process_frame(self, timestamp: float, frame_number: int) -> bool:
        """
        Determine if this frame should be processed based on output frequency.

        Override this method to implement custom frame sampling logic.

        Args:
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            True if the frame should be processed, False otherwise
        """
        # Default: process every frame
        return True

    def cleanup(self) -> None:
        """
        Cleanup resources after processing is complete.

        Override this method to release resources, close files, etc.
        """
        pass

    def get_output_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema describing the output data format.

        This helps with validation and documentation of model outputs.

        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "description": f"Output from {self.get_metadata().name}",
        }
