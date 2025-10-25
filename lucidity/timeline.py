"""Timeline synchronization system for aligning model outputs."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from bisect import bisect_left, bisect_right
import numpy as np
from lucidity.base_model import ModelOutput, OutputType


@dataclass
class TimelineEntry:
    """A single entry in the timeline."""
    timestamp: float
    frame_number: Optional[int]
    model_name: str
    output: ModelOutput


@dataclass
class SynchronizedFrame:
    """A frame with all synchronized model outputs."""
    frame_number: int
    timestamp: float
    outputs: Dict[str, ModelOutput] = field(default_factory=dict)

    def add_output(self, model_name: str, output: ModelOutput) -> None:
        """Add a model output to this frame."""
        self.outputs[model_name] = output

    def has_output(self, model_name: str) -> bool:
        """Check if this frame has output from a specific model."""
        return model_name in self.outputs

    def get_output(self, model_name: str) -> Optional[ModelOutput]:
        """Get output from a specific model."""
        return self.outputs.get(model_name)


class Timeline:
    """
    Timeline for synchronizing outputs from multiple models.

    This class handles alignment of outputs that occur at different frequencies,
    ensuring frame-accurate synchronization with the source video.
    """

    def __init__(self, video_fps: float, total_frames: int):
        """
        Initialize timeline.

        Args:
            video_fps: Frame rate of the source video
            total_frames: Total number of frames in the video
        """
        self.video_fps = video_fps
        self.total_frames = total_frames
        self.frame_duration = 1.0 / video_fps if video_fps > 0 else 0.0

        # Store outputs by model name
        self._outputs: Dict[str, List[TimelineEntry]] = {}

        # Cached synchronized frames
        self._synchronized_cache: Dict[int, SynchronizedFrame] = {}

    def add_output(self, model_name: str, output: ModelOutput) -> None:
        """
        Add a model output to the timeline.

        Args:
            model_name: Name of the model producing the output
            output: Model output to add
        """
        if model_name not in self._outputs:
            self._outputs[model_name] = []

        entry = TimelineEntry(
            timestamp=output.timestamp,
            frame_number=output.frame_number,
            model_name=model_name,
            output=output,
        )

        self._outputs[model_name].append(entry)

        # Invalidate cache for this frame
        if output.frame_number is not None:
            self._synchronized_cache.pop(output.frame_number, None)

    def get_outputs_for_frame(
        self,
        frame_number: int,
        interpolation: str = "nearest"
    ) -> SynchronizedFrame:
        """
        Get all model outputs synchronized to a specific frame.

        Args:
            frame_number: Frame number to synchronize to
            interpolation: Interpolation method ('nearest', 'forward_fill', 'none')
                - 'nearest': Use the output with the closest timestamp
                - 'forward_fill': Use the most recent output before or at this frame
                - 'none': Only use outputs exactly at this frame

        Returns:
            SynchronizedFrame with all available outputs
        """
        # Check cache first
        cache_key = (frame_number, interpolation)
        if frame_number in self._synchronized_cache:
            return self._synchronized_cache[frame_number]

        timestamp = frame_number * self.frame_duration
        sync_frame = SynchronizedFrame(frame_number=frame_number, timestamp=timestamp)

        for model_name, entries in self._outputs.items():
            output = self._find_output_for_timestamp(
                entries, timestamp, frame_number, interpolation
            )
            if output is not None:
                sync_frame.add_output(model_name, output)

        # Cache the result
        self._synchronized_cache[frame_number] = sync_frame
        return sync_frame

    def _find_output_for_timestamp(
        self,
        entries: List[TimelineEntry],
        timestamp: float,
        frame_number: int,
        interpolation: str,
    ) -> Optional[ModelOutput]:
        """
        Find the appropriate output for a given timestamp using interpolation.

        Args:
            entries: List of timeline entries to search
            timestamp: Target timestamp
            frame_number: Target frame number
            interpolation: Interpolation method

        Returns:
            The appropriate ModelOutput or None
        """
        if not entries:
            return None

        if interpolation == "none":
            # Only exact matches
            for entry in entries:
                if entry.frame_number == frame_number:
                    return entry.output
            return None

        elif interpolation == "forward_fill":
            # Use most recent output at or before this timestamp
            best_entry = None
            for entry in entries:
                if entry.timestamp <= timestamp:
                    if best_entry is None or entry.timestamp > best_entry.timestamp:
                        best_entry = entry
            return best_entry.output if best_entry else None

        elif interpolation == "nearest":
            # Use output with closest timestamp
            best_entry = None
            min_diff = float('inf')

            for entry in entries:
                diff = abs(entry.timestamp - timestamp)
                if diff < min_diff:
                    min_diff = diff
                    best_entry = entry

            return best_entry.output if best_entry else None

        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

    def get_all_synchronized_frames(
        self,
        interpolation: str = "nearest"
    ) -> List[SynchronizedFrame]:
        """
        Get all frames with synchronized outputs.

        Args:
            interpolation: Interpolation method to use

        Returns:
            List of synchronized frames
        """
        return [
            self.get_outputs_for_frame(i, interpolation)
            for i in range(self.total_frames)
        ]

    def get_timeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the timeline.

        Returns:
            Dictionary with timeline statistics
        """
        summary = {
            "video_fps": self.video_fps,
            "total_frames": self.total_frames,
            "duration": self.total_frames * self.frame_duration,
            "models": {},
        }

        for model_name, entries in self._outputs.items():
            if entries:
                timestamps = [e.timestamp for e in entries]
                summary["models"][model_name] = {
                    "total_outputs": len(entries),
                    "first_timestamp": min(timestamps),
                    "last_timestamp": max(timestamps),
                    "average_interval": (max(timestamps) - min(timestamps)) / (len(timestamps) - 1)
                    if len(timestamps) > 1 else 0.0,
                }

        return summary

    def get_model_outputs(self, model_name: str) -> List[TimelineEntry]:
        """
        Get all outputs from a specific model.

        Args:
            model_name: Name of the model

        Returns:
            List of timeline entries for this model
        """
        return self._outputs.get(model_name, [])

    def get_output_at_time(
        self,
        model_name: str,
        timestamp: float,
        tolerance: float = 0.001
    ) -> Optional[ModelOutput]:
        """
        Get output from a specific model at a specific time.

        Args:
            model_name: Name of the model
            timestamp: Target timestamp
            tolerance: Maximum time difference to consider a match

        Returns:
            ModelOutput if found within tolerance, None otherwise
        """
        entries = self._outputs.get(model_name, [])

        for entry in entries:
            if abs(entry.timestamp - timestamp) <= tolerance:
                return entry.output

        return None

    def clear_cache(self) -> None:
        """Clear the synchronized frame cache."""
        self._synchronized_cache.clear()

    def get_sparse_frames(
        self,
        model_name: str
    ) -> List[int]:
        """
        Get list of frame numbers where a specific model has outputs.

        Useful for sparse outputs (e.g., scene changes, events).

        Args:
            model_name: Name of the model

        Returns:
            Sorted list of frame numbers with outputs
        """
        entries = self._outputs.get(model_name, [])
        frame_numbers = [
            e.frame_number for e in entries
            if e.frame_number is not None
        ]
        return sorted(set(frame_numbers))
