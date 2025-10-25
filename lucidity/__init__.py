"""Lucidity Preprocessor - Modular video ML inference pipeline."""

__version__ = "0.1.0"
__author__ = "Chris von Csefalvay"

from lucidity.base_model import BaseModel, OutputType, ModelMetadata
from lucidity.masking import (
    CircularMask,
    EndoscopicMaskDetector,
    detect_mask_from_video,
)

__all__ = [
    "BaseModel",
    "OutputType",
    "ModelMetadata",
    "CircularMask",
    "EndoscopicMaskDetector",
    "detect_mask_from_video",
]
