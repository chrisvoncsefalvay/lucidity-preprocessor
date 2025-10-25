"""
Test script for endoscopic masking functionality.

This script demonstrates how to:
1. Detect circular masks from endoscopic videos
2. Visualise the detected mask
3. Apply masks to video frames
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path to import lucidity
sys.path.insert(0, str(Path(__file__).parent.parent))

from lucidity.masking import detect_mask_from_video, EndoscopicMaskDetector


def visualise_mask_detection(video_path: str, output_path: str = None, n_frames: int = 10):
    """
    Visualise the mask detection process.

    Args:
        video_path: Path to input video
        output_path: Optional path to save visualisation
        n_frames: Number of frames to analyse
    """
    print(f"Detecting mask from video: {video_path}")
    print(f"Analysing first {n_frames} frames...")

    # Detect mask
    mask = detect_mask_from_video(video_path, n_frames=n_frames)

    if mask is None:
        print("Failed to detect mask!")
        return

    print(f"Mask detected successfully!")
    print(f"  Centre: {mask.centre}")
    print(f"  Radius: {mask.radius:.2f}")

    # Open video to get a sample frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read sample frame")
        return

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create visualisation
    h, w = frame_rgb.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)

    # Original frame
    vis[:, :w] = frame_rgb

    # Mask visualisation
    mask_vis = np.stack([mask.mask * 255] * 3, axis=-1)
    vis[:, w:2*w] = mask_vis

    # Masked frame
    masked_frame = mask.apply(frame_rgb)
    vis[:, 2*w:] = masked_frame

    # Draw circle on all three panels
    centre_x, centre_y = mask.centre
    radius = int(mask.radius)
    colour = (0, 255, 0)  # Green

    for offset in [0, w, 2*w]:
        cv2.circle(vis, (centre_x + offset, centre_y), radius, colour, 2)
        cv2.circle(vis, (centre_x + offset, centre_y), 3, colour, -1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_colour = (255, 255, 255)
    font_thickness = 2

    cv2.putText(vis, "Original", (10, 30), font, font_scale, font_colour, font_thickness)
    cv2.putText(vis, "Detected Mask", (w + 10, 30), font, font_scale, font_colour, font_thickness)
    cv2.putText(vis, "Masked Frame", (2*w + 10, 30), font, font_scale, font_colour, font_thickness)

    # Save or display
    if output_path:
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        print(f"Visualisation saved to: {output_path}")
    else:
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow("Mask Detection", vis_bgr)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def apply_mask_to_video(
    video_path: str,
    output_path: str,
    n_frames: int = 10,
    max_frames: int = None
):
    """
    Apply detected mask to entire video and save result.

    Args:
        video_path: Path to input video
        output_path: Path to save masked video
        n_frames: Number of frames to analyse for mask detection
        max_frames: Maximum frames to process (None for all)
    """
    print(f"Processing video: {video_path}")

    # Detect mask
    mask = detect_mask_from_video(video_path, n_frames=n_frames)

    if mask is None:
        print("Failed to detect mask!")
        return

    print(f"Mask detected: centre={mask.centre}, radius={mask.radius:.2f}")

    # Open input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Input video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Processing frames...")

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply mask
        masked_rgb = mask.apply(frame_rgb)

        # Convert back to BGR for video writer
        masked_bgr = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)

        # Write frame
        out.write(masked_bgr)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()

    print(f"Masked video saved to: {output_path}")
    print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Test endoscopic masking functionality"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input endoscopic video"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["visualise", "process"],
        default="visualise",
        help="Mode: 'visualise' shows mask detection, 'process' creates masked video"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (image for visualise mode, video for process mode)"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of frames to analyse for mask detection (default: 10)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process in process mode (default: all)"
    )

    args = parser.parse_args()

    if args.mode == "visualise":
        visualise_mask_detection(
            args.video_path,
            args.output,
            args.n_frames
        )
    elif args.mode == "process":
        if not args.output:
            args.output = str(Path(args.video_path).with_stem(
                Path(args.video_path).stem + "_masked"
            ))

        apply_mask_to_video(
            args.video_path,
            args.output,
            args.n_frames,
            args.max_frames
        )


if __name__ == "__main__":
    main()
