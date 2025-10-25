"""Endoscopic video masking utilities.

This module provides functionality to detect and apply circular masks to endoscopic videos,
where the actual endoscopic image appears as a circular region on a black background.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CircularMask:
    """Represents a circular mask for endoscopic video frames."""
    mask: np.ndarray  # Binary mask (H, W) where 1 = valid region, 0 = black border
    centre: Tuple[int, int]  # (x, y) coordinates of circle centre
    radius: float  # Radius of the circular region

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the mask to a frame, setting masked regions to black.

        Args:
            frame: Input frame (H, W, C) or (H, W)

        Returns:
            Masked frame with same shape as input
        """
        if len(frame.shape) == 3:
            # Multi-channel image
            return frame * self.mask[:, :, np.newaxis]
        else:
            # Single channel
            return frame * self.mask


class EndoscopicMaskDetector:
    """
    Detects circular masks in endoscopic videos.

    Analyses the first N frames to identify the circular region containing
    the actual endoscopic image, separating it from the black border/background.
    """

    def __init__(
        self,
        n_frames: int = 10,
        black_threshold: int = 30,
        min_valid_ratio: float = 0.3,
        morph_kernel_size: int = 5,
        circle_fit_method: str = 'hough'
    ):
        """
        Initialise the mask detector.

        Args:
            n_frames: Number of initial frames to analyse
            black_threshold: Pixel intensity threshold for considering a pixel black (0-255)
            min_valid_ratio: Minimum ratio of frames where a pixel must be non-black
                           to be considered part of the valid region
            morph_kernel_size: Kernel size for morphological operations
            circle_fit_method: Method for fitting circle ('hough' or 'contour')
        """
        self.n_frames = n_frames
        self.black_threshold = black_threshold
        self.min_valid_ratio = min_valid_ratio
        self.morph_kernel_size = morph_kernel_size
        self.circle_fit_method = circle_fit_method

    def detect_mask_from_frames(self, frames: np.ndarray) -> CircularMask:
        """
        Detect the circular mask from a sequence of frames.

        Args:
            frames: Array of frames with shape (N, H, W, C) or (N, H, W)

        Returns:
            CircularMask object representing the detected circular region
        """
        n, h, w = frames.shape[:3]

        # Convert to greyscale if needed
        if len(frames.shape) == 4:
            grey_frames = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
        else:
            grey_frames = frames

        # Count how many frames each pixel is non-black
        valid_pixel_count = np.sum(grey_frames > self.black_threshold, axis=0)

        # Create initial binary mask based on threshold
        initial_mask = (valid_pixel_count >= (n * self.min_valid_ratio)).astype(np.uint8)

        # Apply morphological operations to clean up the mask
        cleaned_mask = self._clean_mask(initial_mask)

        # Fit a circle to the mask
        centre, radius = self._fit_circle_to_mask(cleaned_mask)

        # Create final circular mask
        final_mask = self._create_circular_mask(h, w, centre, radius)

        return CircularMask(mask=final_mask, centre=centre, radius=radius)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up binary mask using morphological operations.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Cleaned binary mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )

        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill remaining holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        return mask

    def _fit_circle_to_mask(self, mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Fit a circle to the binary mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Tuple of (centre, radius) where centre is (x, y)
        """
        if self.circle_fit_method == 'hough':
            return self._fit_circle_hough(mask)
        elif self.circle_fit_method == 'contour':
            return self._fit_circle_contour(mask)
        else:
            raise ValueError(f"Unknown circle fit method: {self.circle_fit_method}")

    def _fit_circle_hough(self, mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Fit circle using Hough Circle Transform.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Tuple of (centre, radius) where centre is (x, y)
        """
        # Convert to uint8 for Hough transform
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Detect edges
        edges = cv2.Canny(mask_uint8, 50, 150)

        # Detect circles
        h, w = mask.shape
        min_radius = int(min(h, w) * 0.2)
        max_radius = int(min(h, w) * 0.6)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(h, w),
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None and len(circles[0]) > 0:
            # Take the first (strongest) circle
            x, y, r = circles[0][0]
            return (int(x), int(y)), float(r)
        else:
            # Fallback to contour method if Hough fails
            return self._fit_circle_contour(mask)

    def _fit_circle_contour(self, mask: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Fit circle using contour analysis and minimum enclosing circle.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Tuple of (centre, radius) where centre is (x, y)
        """
        # Find contours
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback to image centre if no contours found
            h, w = mask.shape
            return (w // 2, h // 2), min(h, w) // 2

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        return (int(x), int(y)), float(radius)

    def _create_circular_mask(
        self,
        height: int,
        width: int,
        centre: Tuple[int, int],
        radius: float
    ) -> np.ndarray:
        """
        Create a perfect circular mask.

        Args:
            height: Mask height
            width: Mask width
            centre: Circle centre (x, y)
            radius: Circle radius

        Returns:
            Binary mask with perfect circular region
        """
        y, x = np.ogrid[:height, :width]
        cx, cy = centre

        # Calculate distance from centre
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Create circular mask with anti-aliasing
        mask = (distance <= radius).astype(np.uint8)

        # Apply slight Gaussian blur and threshold for smooth edges
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
        mask = (mask > 0.5).astype(np.uint8)

        return mask


def detect_mask_from_video(
    video_path: str,
    n_frames: int = 10,
    **detector_kwargs
) -> Optional[CircularMask]:
    """
    Detect circular mask from a video file.

    Args:
        video_path: Path to video file
        n_frames: Number of frames to analyse
        **detector_kwargs: Additional arguments for EndoscopicMaskDetector

    Returns:
        CircularMask object or None if detection fails
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_count = 0

    while frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1

    cap.release()

    if not frames:
        return None

    frames_array = np.array(frames)

    detector = EndoscopicMaskDetector(n_frames=len(frames), **detector_kwargs)
    return detector.detect_mask_from_frames(frames_array)
