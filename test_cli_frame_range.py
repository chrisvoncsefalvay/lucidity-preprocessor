"""
Integration test for CLI frame range functionality with example model.
"""

import cv2
import numpy as np
import tempfile
import os
import shutil
import subprocess
import json


def create_test_video(filename: str, num_frames: int = 100, fps: int = 30):
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


def test_cli_with_frame_range():
    """Test CLI processing with frame range."""
    print("Testing CLI with frame range...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_test_video(video_path, num_frames=100, fps=30)
        print(f"Created test video: {video_path}")

        # Test 1: Process frames 10-19 (10 frames)
        print("\nTest 1: Processing frames 10-19")
        output_dir = os.path.join(tmpdir, "output_frames")
        result = subprocess.run([
            "./venv/Scripts/lucidity.exe", "process",
            video_path,
            "--models", "example_frame_model",
            "--discover-dir", "./examples",
            "--output", output_dir,
            "--start-frame", "10",
            "--end-frame", "19",
            "--no-progress",
        ], capture_output=True, text=True)

        print("Output:", result.stdout)
        if result.returncode != 0:
            print("Error:", result.stderr)
            return False

        # Check manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        frame_range = manifest.get("processing_info", {}).get("frame_range", {})
        print(f"  Frame range in manifest: {frame_range}")

        assert frame_range["start_frame"] == 10, "Start frame should be 10"
        assert frame_range["end_frame"] == 19, "End frame should be 19"
        assert frame_range["total_frames_processed"] == 10, "Should process 10 frames"

        # Test 2: Process with time range (first 1 second = 30 frames at 30fps)
        print("\nTest 2: Processing 0-1 second")
        output_dir2 = os.path.join(tmpdir, "output_time")
        result = subprocess.run([
            "./venv/Scripts/lucidity.exe", "process",
            video_path,
            "--models", "example_frame_model",
            "--discover-dir", "./examples",
            "--output", output_dir2,
            "--start-time", "0",
            "--end-time", "1",
            "--no-progress",
        ], capture_output=True, text=True)

        print("Output:", result.stdout)
        if result.returncode != 0:
            print("Error:", result.stderr)
            return False

        # Check manifest
        manifest_path2 = os.path.join(output_dir2, "manifest.json")
        with open(manifest_path2, 'r') as f:
            manifest2 = json.load(f)

        frame_range2 = manifest2.get("processing_info", {}).get("frame_range", {})
        print(f"  Frame range in manifest: {frame_range2}")

        # At 30fps, 0-1 second should be frames 0-29 (30 frames)
        assert frame_range2["start_frame"] == 0, "Start frame should be 0"
        assert frame_range2["end_frame"] == 30, "End frame should be 30"
        assert frame_range2["total_frames_processed"] == 31, "Should process 31 frames (0-30 inclusive)"

        print("\nAll CLI integration tests passed!")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("CLI Integration Test with Frame Range")
    print("=" * 60)

    success = test_cli_with_frame_range()

    print("\n" + "=" * 60)
    if success:
        print("Integration tests completed successfully!")
    else:
        print("Integration tests failed!")
    print("=" * 60)
