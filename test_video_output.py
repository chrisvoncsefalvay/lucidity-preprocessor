"""
Test script for video output functionality with endomust_depth model.
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


def test_video_output_format():
    """Test video output format with endomust_depth."""
    print("Testing video output format with endomust_depth model...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test video (10 frames)
        video_path = os.path.join(tmpdir, "test_video.mp4")
        create_test_video(video_path, num_frames=10, fps=30)
        print(f"Created test video: {video_path}")

        # Test 1: Default output (individual frames)
        print("\n" + "=" * 60)
        print("Test 1: Default output (individual .npy frames)")
        print("=" * 60)
        output_dir1 = os.path.join(tmpdir, "output_frames")
        result1 = subprocess.run([
            "./venv/Scripts/lucidity.exe", "process",
            video_path,
            "--models", "endomust_depth",
            "--output", output_dir1,
            "--start-frame", "0",
            "--end-frame", "4",
            "--no-progress",
        ], capture_output=True, text=True)

        print(result1.stdout)
        if result1.returncode != 0:
            print("STDERR:", result1.stderr)
            return False

        # Check output files
        frames_dir = os.path.join(output_dir1, "endomust_depth", "frames")
        if os.path.exists(frames_dir):
            num_files = len([f for f in os.listdir(frames_dir) if f.endswith('.npy')])
            print(f"Generated {num_files} .npy files")
        else:
            print("ERROR: Frames directory not found!")
            return False

        # Test 2: Video output format
        print("\n" + "=" * 60)
        print("Test 2: Video output format")
        print("=" * 60)
        output_dir2 = os.path.join(tmpdir, "output_video")
        result2 = subprocess.run([
            "./venv/Scripts/lucidity.exe", "process",
            video_path,
            "--models", "endomust_depth",
            "--output", output_dir2,
            "--start-frame", "0",
            "--end-frame", "4",
            "--output-format", "video",
            "--fps", "30",
            "--no-progress",
        ], capture_output=True, text=True)

        print(result2.stdout)
        if result2.returncode != 0:
            print("STDERR:", result2.stderr)
            return False

        # Check video file
        video_file = os.path.join(output_dir2, "endomust_depth", "output.mp4")
        if os.path.exists(video_file):
            file_size = os.path.getsize(video_file) / 1024  # KB
            print(f"Generated video file: {video_file}")
            print(f"Video file size: {file_size:.2f} KB")

            # Try to read the video to verify it's valid
            cap = cv2.VideoCapture(video_file)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            cap.release()

            print(f"Video contains {frame_count} frames")

            if frame_count != 5:
                print(f"WARNING: Expected 5 frames, got {frame_count}")

        else:
            print("ERROR: Video file not found!")
            return False

        print("\nAll tests passed!")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Video Output Format Test")
    print("=" * 60)

    success = test_video_output_format()

    print("\n" + "=" * 60)
    if success:
        print("Video output test completed successfully!")
    else:
        print("Video output test failed!")
    print("=" * 60)
