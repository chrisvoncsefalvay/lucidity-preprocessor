"""
Test script for frame range functionality.
"""

import cv2
import numpy as np
import tempfile
import os
from pathlib import Path


def create_test_video(filename: str, num_frames: int = 100, fps: int = 30):
    """Create a simple test video with frame numbers displayed."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create frame with different colours based on frame number
        colour = (
            (i * 255 // num_frames) % 256,
            ((i * 127) % 256),
            ((255 - i * 255 // num_frames) % 256)
        )

        frame = np.full((height, width, 3), colour, dtype=np.uint8)

        # Add frame number text
        text = f"Frame {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (50, 240), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Created test video: {filename} ({num_frames} frames, {fps} fps)")


def test_frame_range_iteration():
    """Test VideoReader frame range iteration."""
    from lucidity.video import VideoReader

    # Create test video
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_test_video(video_path, num_frames=100, fps=30)

        # Test 1: Read all frames
        print("\nTest 1: Reading all frames")
        reader = VideoReader(video_path)
        frames_read = 0
        for _, _, frame_num in reader.frames():
            frames_read += 1
        print(f"  Read {frames_read} frames (expected 100)")
        assert frames_read == 100, "Should read all 100 frames"
        reader.close()

        # Test 2: Read frames 20-29
        print("\nTest 2: Reading frames 20-29")
        reader = VideoReader(video_path)
        frame_numbers = []
        for _, _, frame_num in reader.frames(start_frame=20, end_frame=29):
            frame_numbers.append(frame_num)
        print(f"  Frame numbers: {frame_numbers}")
        assert frame_numbers == list(range(20, 30)), "Should read frames 20-29"
        reader.close()

        # Test 3: Read first 10 frames
        print("\nTest 3: Reading first 10 frames")
        reader = VideoReader(video_path)
        frame_numbers = []
        for _, _, frame_num in reader.frames(start_frame=0, end_frame=9):
            frame_numbers.append(frame_num)
        print(f"  Frame numbers: {frame_numbers}")
        assert frame_numbers == list(range(0, 10)), "Should read frames 0-9"
        reader.close()

        # Test 4: Read last 10 frames
        print("\nTest 4: Reading last 10 frames")
        reader = VideoReader(video_path)
        frame_numbers = []
        for _, _, frame_num in reader.frames(start_frame=90, end_frame=99):
            frame_numbers.append(frame_num)
        print(f"  Frame numbers: {frame_numbers}")
        assert frame_numbers == list(range(90, 100)), "Should read frames 90-99"
        reader.close()

        # Test 5: Test timestamps
        print("\nTest 5: Checking timestamps")
        reader = VideoReader(video_path)
        for frame, timestamp, frame_num in reader.frames(start_frame=30, end_frame=30):
            expected_timestamp = frame_num / reader.metadata.fps
            print(f"  Frame {frame_num}: timestamp={timestamp:.3f}s (expected {expected_timestamp:.3f}s)")
            assert abs(timestamp - expected_timestamp) < 0.001, "Timestamp should match frame number"
        reader.close()

    print("\nAll tests passed!")


def test_cli_help():
    """Test that CLI help shows new options."""
    import subprocess

    print("\nTesting CLI help output...")
    result = subprocess.run(
        ["./venv/Scripts/lucidity.exe", "process", "--help"],
        capture_output=True,
        text=True
    )

    help_text = result.stdout

    # Check for new options
    assert "--start-time" in help_text, "Should show --start-time option"
    assert "--end-time" in help_text, "Should show --end-time option"
    assert "--start-frame" in help_text, "Should show --start-frame option"
    assert "--end-frame" in help_text, "Should show --end-frame option"

    print("CLI help shows new options:")
    print("  - --start-time")
    print("  - --end-time")
    print("  - --start-frame")
    print("  - --end-frame")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Frame Range Functionality")
    print("=" * 60)

    test_frame_range_iteration()
    test_cli_help()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
