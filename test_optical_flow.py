"""
Test script for RAFT optical flow model.

This script tests the optical flow implementation on a sample video.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lucidity.plugin_manager import PluginManager
from lucidity.processor import VideoProcessor


def test_optical_flow():
    """Test optical flow on a sample video."""

    # Check if test video exists
    test_videos = [
        r"D:\datasets\cholec80\videos\video01.mp4",
        r"D:\datasets\cholec80\videos\video02.mp4",
    ]

    test_video = None
    for video_path in test_videos:
        if os.path.exists(video_path):
            test_video = video_path
            break

    if test_video is None:
        print("Error: No test video found")
        print("Please provide a valid video path")
        return

    print(f"Testing optical flow on: {test_video}")
    print("-" * 80)

    # Create plugin manager
    plugin_manager = PluginManager()

    # Discover models from the models directory
    models_dir = Path(__file__).parent / "models"
    plugin_manager.discover_from_directory(models_dir)

    # List available models
    print("\nAvailable models:")
    for name in plugin_manager.list_plugins():
        print(f"  - {name}")

    # Create processor
    output_dir = "./test_optical_flow_output"
    processor = VideoProcessor(
        video_path=test_video,
        output_dir=output_dir,
        plugin_manager=plugin_manager,
    )

    # Add optical flow model
    print("\nAdding RAFT optical flow model...")
    processor.add_model("raft_optical_flow", config={
        "model_size": "small",  # Use small model for faster processing
        "num_flow_updates": 12,
    })

    # Process first 30 frames
    print("\nProcessing first 30 frames...")
    manifest_path = processor.process(
        show_progress=True,
        start_frame=0,
        end_frame=29,
        output_format='compressed',  # Use compressed format for efficient storage
    )

    print(f"\nProcessing complete!")
    print(f"Manifest: {manifest_path}")
    print(f"Output directory: {output_dir}")

    # Print compression statistics
    import numpy as np
    from lucidity.manifest import load_manifest

    manifest = load_manifest(manifest_path)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for model_name, model_output in manifest.model_outputs.items():
        print(f"\nModel: {model_name}")
        print(f"  Frames processed: {model_output.frame_coverage.frames_processed}")
        print(f"  Output files:")
        for output_file in model_output.output_files:
            print(f"    - {output_file.path}")
            print(f"      Type: {output_file.type}")
            print(f"      Format: {output_file.format}")
            if output_file.size_bytes:
                size_mb = output_file.size_bytes / (1024 * 1024)
                print(f"      Size: {size_mb:.2f} MB")

    # Calculate compression ratio if possible
    if manifest.model_outputs:
        for model_name, model_output in manifest.model_outputs.items():
            if model_output.output_files:
                npz_file = None
                for output_file in model_output.output_files:
                    if output_file.format == "npz":
                        npz_file = Path(output_dir) / output_file.path
                        break

                if npz_file and npz_file.exists():
                    # Load and calculate uncompressed size
                    with np.load(npz_file) as data:
                        if 'frames' in data:
                            frames = data['frames']
                            uncompressed_size = frames.nbytes
                            compressed_size = npz_file.stat().st_size
                            ratio = uncompressed_size / compressed_size

                            print(f"\n  Compression statistics:")
                            print(f"    Uncompressed size: {uncompressed_size / (1024 * 1024):.2f} MB")
                            print(f"    Compressed size: {compressed_size / (1024 * 1024):.2f} MB")
                            print(f"    Compression ratio: {ratio:.2f}x")


if __name__ == '__main__':
    test_optical_flow()
