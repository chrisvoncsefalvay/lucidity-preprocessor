"""Video processing and metadata extraction."""

import cv2
from pathlib import Path
from typing import Generator, Tuple, Optional
import numpy as np
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # in seconds
    codec: str

    @property
    def frame_duration(self) -> float:
        """Duration of a single frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """
    Video reader with frame-accurate timing information.

    This class provides a consistent interface for reading video frames
    with precise timing information for synchronization.
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from the video file."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Get codec information
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        return VideoMetadata(
            path=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        return self._metadata

    def frames(self) -> Generator[Tuple[np.ndarray, float, int], None, None]:
        """
        Generate frames with timing information.

        Yields:
            Tuple of (frame, timestamp, frame_number)
            - frame: RGB numpy array (H, W, 3)
            - timestamp: Time in seconds from video start
            - frame_number: Frame number (0-indexed)
        """
        frame_number = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Calculate timestamp
            timestamp = frame_number * self._metadata.frame_duration

            yield frame_rgb, timestamp, frame_number
            frame_number += 1

    def get_frame_at_time(self, timestamp: float) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get frame at a specific timestamp.

        Args:
            timestamp: Time in seconds from video start

        Returns:
            Tuple of (frame, actual_timestamp, frame_number) or None if failed
        """
        frame_number = int(timestamp * self._metadata.fps)
        if frame_number >= self._metadata.total_frames:
            return None

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        actual_timestamp = frame_number * self._metadata.frame_duration

        return frame_rgb, actual_timestamp, frame_number

    def get_frame_by_number(self, frame_number: int) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Get frame by frame number.

        Args:
            frame_number: Frame number (0-indexed)

        Returns:
            Tuple of (frame, timestamp, frame_number) or None if failed
        """
        if frame_number >= self._metadata.total_frames or frame_number < 0:
            return None

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = frame_number * self._metadata.frame_duration

        return frame_rgb, timestamp, frame_number

    def reset(self) -> None:
        """Reset video to the beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        """Close the video file."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure video is closed."""
        self.close()
