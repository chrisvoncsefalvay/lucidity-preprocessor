"""Manifest generation for processed video outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import numpy as np

from lucidity.video import VideoMetadata
from lucidity.base_model import ModelMetadata, OutputType


class OutputFileInfo(BaseModel):
    """Information about an output file."""
    path: str
    type: str  # 'frames', 'data', 'metadata'
    format: str  # 'png', 'npy', 'json', etc.
    description: str
    size_bytes: Optional[int] = None


class ModelOutputManifest(BaseModel):
    """Manifest for outputs from a single model."""
    model_metadata: ModelMetadata
    output_files: List[OutputFileInfo] = Field(default_factory=list)
    total_outputs: int
    first_timestamp: float
    last_timestamp: float
    output_frequency_hz: Optional[float] = None
    frame_coverage: Dict[str, Any] = Field(default_factory=dict)  # frame ranges with outputs


class ProcessingManifest(BaseModel):
    """Complete manifest for a processed video."""
    version: str = "1.0"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    video_metadata: VideoMetadata
    models: Dict[str, ModelOutputManifest] = Field(default_factory=dict)
    output_directory: str
    processing_time_seconds: Optional[float] = None
    timeline_summary: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }


class ManifestBuilder:
    """Builder for creating processing manifests."""

    def __init__(self, video_metadata: VideoMetadata, output_dir: Path):
        """
        Initialize manifest builder.

        Args:
            video_metadata: Metadata from the source video
            output_dir: Directory where outputs are stored
        """
        self.video_metadata = video_metadata
        self.output_dir = output_dir
        self.model_manifests: Dict[str, ModelOutputManifest] = {}
        self.timeline_summary: Dict[str, Any] = {}
        self.processing_start_time: Optional[datetime] = None
        self.processing_end_time: Optional[datetime] = None

    def start_processing(self) -> None:
        """Mark the start of processing."""
        self.processing_start_time = datetime.now()

    def end_processing(self) -> None:
        """Mark the end of processing."""
        self.processing_end_time = datetime.now()

    def add_model(
        self,
        model_metadata: ModelMetadata,
        output_files: List[OutputFileInfo],
        total_outputs: int,
        first_timestamp: float,
        last_timestamp: float,
        frame_coverage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a model's output information to the manifest.

        Args:
            model_metadata: Metadata about the model
            output_files: List of files produced by this model
            total_outputs: Total number of outputs produced
            first_timestamp: Timestamp of first output
            last_timestamp: Timestamp of last output
            frame_coverage: Information about which frames have outputs
        """
        # Calculate output frequency
        duration = last_timestamp - first_timestamp
        output_frequency_hz = None
        if duration > 0 and total_outputs > 1:
            output_frequency_hz = (total_outputs - 1) / duration

        model_manifest = ModelOutputManifest(
            model_metadata=model_metadata,
            output_files=output_files,
            total_outputs=total_outputs,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            output_frequency_hz=output_frequency_hz,
            frame_coverage=frame_coverage or {},
        )

        self.model_manifests[model_metadata.name] = model_manifest

    def set_timeline_summary(self, summary: Dict[str, Any]) -> None:
        """
        Set the timeline summary.

        Args:
            summary: Timeline summary from Timeline.get_timeline_summary()
        """
        self.timeline_summary = summary

    def build(self) -> ProcessingManifest:
        """
        Build the complete manifest.

        Returns:
            ProcessingManifest object
        """
        processing_time = None
        if self.processing_start_time and self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            processing_time = delta.total_seconds()

        manifest = ProcessingManifest(
            video_metadata=self.video_metadata,
            models=self.model_manifests,
            output_directory=str(self.output_dir),
            processing_time_seconds=processing_time,
            timeline_summary=self.timeline_summary,
        )

        return manifest

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the manifest to a JSON file.

        Args:
            filepath: Path to save the manifest. If None, saves to output_dir/manifest.json

        Returns:
            Path to the saved manifest file
        """
        if filepath is None:
            filepath = self.output_dir / "manifest.json"

        manifest = self.build()

        # Convert to dict and handle numpy arrays
        manifest_dict = json.loads(manifest.json(indent=2))

        with open(filepath, 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        return filepath


def load_manifest(filepath: Path) -> ProcessingManifest:
    """
    Load a manifest from a JSON file.

    Args:
        filepath: Path to the manifest file

    Returns:
        ProcessingManifest object
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return ProcessingManifest(**data)


def create_output_package_summary(manifest: ProcessingManifest) -> str:
    """
    Create a human-readable summary of the output package.

    Args:
        manifest: Processing manifest

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 60,
        "Lucidity Processing Summary",
        "=" * 60,
        f"Video: {manifest.video_metadata.path}",
        f"Duration: {manifest.video_metadata.duration:.2f}s",
        f"Frames: {manifest.video_metadata.total_frames} @ {manifest.video_metadata.fps:.2f} fps",
        f"Resolution: {manifest.video_metadata.width}x{manifest.video_metadata.height}",
        "",
        f"Models processed: {len(manifest.models)}",
        "",
    ]

    for model_name, model_manifest in manifest.models.items():
        lines.extend([
            f"Model: {model_name}",
            f"  Output type: {model_manifest.model_metadata.output_type}",
            f"  Total outputs: {model_manifest.total_outputs}",
        ])

        if model_manifest.output_frequency_hz:
            lines.append(f"  Frequency: {model_manifest.output_frequency_hz:.2f} Hz")

        lines.append(f"  Files: {len(model_manifest.output_files)}")
        for file_info in model_manifest.output_files:
            size_str = f" ({file_info.size_bytes / 1024 / 1024:.2f} MB)" if file_info.size_bytes else ""
            lines.append(f"    - {file_info.path}{size_str}")
        lines.append("")

    if manifest.processing_time_seconds:
        lines.append(f"Processing time: {manifest.processing_time_seconds:.2f}s")

    lines.append("=" * 60)

    return "\n".join(lines)
