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

    # Add optical flow model with sparse configuration
    print("\nAdding RAFT sparse optical flow model...")
    processor.add_model("raft_optical_flow", config={
        "model_size": "small",  # Use small model for faster processing
        "num_flow_updates": 12,
        "stride": 16,  # Sample every 16 pixels
        "mask_threshold": 30,
        "mask_method": "hough",
    })

    # Process first 30 frames
    print("\nProcessing first 30 frames...")
    manifest_path = processor.process(
        show_progress=True,
        start_frame=0,
        end_frame=29,
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

    # Calculate storage efficiency for sparse flow
    if manifest.model_outputs:
        for model_name, model_output in manifest.model_outputs.items():
            if model_output.output_files:
                json_file = None
                for output_file in model_output.output_files:
                    if output_file.format == "json":
                        json_file = Path(output_dir) / output_file.path
                        break

                if json_file and json_file.exists():
                    # Load sparse flow data
                    import json
                    with open(json_file, 'r') as f:
                        flow_data = json.load(f)

                    # Count total vectors
                    total_vectors = sum(len(entry['data']['x']) for entry in flow_data)
                    avg_vectors_per_frame = total_vectors / len(flow_data) if flow_data else 0

                    # Estimate what dense flow would have cost
                    # Assuming 1920x1080 resolution, stride=16 would give ~8100 vectors per frame
                    # Dense flow would be 1920*1080*2*4 bytes = 16.6 MB per frame
                    from lucidity.video import VideoReader
                    video_reader = VideoReader(test_video)
                    h, w = video_reader.metadata.height, video_reader.metadata.width
                    dense_size_per_frame = h * w * 2 * 4  # 2 channels, 4 bytes per float32
                    dense_total_size = dense_size_per_frame * len(flow_data)

                    # Actual sparse size
                    sparse_size = json_file.stat().st_size

                    # Calculate reduction
                    reduction_ratio = dense_total_size / sparse_size if sparse_size > 0 else 0

                    print(f"\n  Storage efficiency:")
                    print(f"    Average vectors per frame: {avg_vectors_per_frame:.0f}")
                    print(f"    Total vectors: {total_vectors}")
                    print(f"    Dense equivalent size: {dense_total_size / (1024 * 1024):.2f} MB")
                    print(f"    Sparse storage size: {sparse_size / (1024 * 1024):.2f} MB")
                    print(f"    Storage reduction: {reduction_ratio:.2f}x")

                    # Test visualization
                    if flow_data and len(flow_data) > 0:
                        print(f"\n  Creating sample visualisations...")
                        from lucidity.flow_plotting import render_sparse_flow_on_frame

                        # Get middle frame
                        mid_idx = len(flow_data) // 2
                        mid_frame_num = flow_data[mid_idx]['frame_number']

                        # Read that frame
                        video_reader = VideoReader(test_video)
                        for frame_info in video_reader.read_frames(start_frame=mid_frame_num, end_frame=mid_frame_num):
                            frame = frame_info.frame
                            sparse_flow_data = flow_data[mid_idx]['data']

                            # Render flow on frame
                            vis_frame = render_sparse_flow_on_frame(
                                sparse_flow_data,
                                frame,
                                scale=2.0,
                                color=(0, 255, 255),  # Cyan
                                thickness=2,
                            )

                            # Save visualization
                            import cv2
                            vis_path = Path(output_dir) / "flow_visualization_sample.png"
                            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                            print(f"    Saved visualisation: {vis_path}")


if __name__ == '__main__':
    test_optical_flow()
