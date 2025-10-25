"""
Quick test to verify endomust_depth model works with CLI.
"""

import cv2
import numpy as np
import tempfile
import os
import subprocess


def create_test_video(filename: str, num_frames: int = 10, fps: int = 30):
    """Create a simple test video."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(num_frames):
        colour = (
            (i * 255 // num_frames) % 256,
            ((i * 127) % 256),
            ((255 - i * 255 // num_frames) % 256)
        )
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        text = f"Frame {i}"
        cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)

    out.release()


def test_endomust_cli():
    """Test endomust_depth model with CLI."""
    print("Testing endomust_depth model with CLI...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video (only 5 frames to be fast)
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_test_video(video_path, num_frames=5, fps=30)
        print(f"Created test video: {video_path}")

        # Process with endomust_depth model (only first 3 frames)
        output_dir = os.path.join(tmpdir, "output")
        print("\nProcessing with endomust_depth model...")
        result = subprocess.run([
            "./venv/Scripts/lucidity.exe", "process",
            video_path,
            "--models", "endomust_depth",
            "--output", output_dir,
            "--start-frame", "0",
            "--end-frame", "2",
            "--no-progress",
        ], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.returncode != 0:
            print("\nSTDERR:")
            print(result.stderr)
            print("\nTest FAILED!")
            return False
        else:
            print("\nTest PASSED!")
            print(f"Output saved to: {output_dir}")
            return True


if __name__ == "__main__":
    print("=" * 60)
    print("EndoMUST CLI Test")
    print("=" * 60)

    success = test_endomust_cli()

    print("\n" + "=" * 60)
    if success:
        print("CLI test completed successfully!")
    else:
        print("CLI test failed!")
    print("=" * 60)
