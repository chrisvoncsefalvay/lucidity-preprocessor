"""
Test script for the ConvLSTM Surgical Tool Tracker model.

This script tests the tool tracking model integration with Lucidity.
"""

import sys
from pathlib import Path

# Add models directory to path
models_dir = Path(__file__).parent / 'models'
sys.path.insert(0, str(models_dir))

from lucidity.plugin_manager import PluginManager
from lucidity.processor import VideoProcessor


def test_tool_tracking_model():
    """Test loading and initialising the tool tracking model."""
    print("Testing ConvLSTM Surgical Tool Tracker...")
    print("=" * 60)

    # Get plugin manager
    plugin_manager = PluginManager()

    # Discover models from the models directory
    plugin_manager.discover_from_directory(models_dir)

    # Check if tool_tracking is available
    available_models = plugin_manager.list_plugins()
    print(f"\nAvailable models: {', '.join(available_models)}")

    if 'tool_tracking' not in available_models:
        print("\nError: tool_tracking model not found!")
        print("Make sure tool_tracking_model.py is in models/tool_tracking/")
        return False

    print("\nModel found: tool_tracking")

    # Get model metadata
    try:
        metadata = plugin_manager.get_plugin_metadata('tool_tracking')
        print("\nModel Metadata:")
        print(f"  Name: {metadata.name}")
        print(f"  Version: {metadata.version}")
        print(f"  Description: {metadata.description}")
        print(f"  Output Type: {metadata.output_type}")
        print(f"  Output Frequency: {metadata.output_frequency}")
        print(f"  Dependencies: {', '.join(metadata.dependencies)}")
    except Exception as e:
        print(f"\nError getting metadata: {e}")
        return False

    # Try to initialise the model
    print("\nAttempting to initialise model...")
    try:
        model = plugin_manager.get_plugin('tool_tracking')
        model.initialize()
        print("Model initialised successfully!")

        # Cleanup
        model.cleanup()
        print("Model cleaned up successfully!")

        return True
    except Exception as e:
        print(f"\nError initialising model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_video(video_path: str):
    """
    Test the tool tracking model with an actual video.

    Args:
        video_path: Path to a surgical video file
    """
    print(f"\nTesting with video: {video_path}")
    print("=" * 60)

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return False

    try:
        # Get plugin manager
        plugin_manager = PluginManager()
        plugin_manager.discover_from_directory(Path(__file__).parent / 'models')

        # Create processor
        processor = VideoProcessor(
            video_path=video_path,
            output_dir='./test_output/tool_tracking',
            plugin_manager=plugin_manager,
        )

        # Add model
        processor.add_model('tool_tracking')

        # Process (just first 10 frames for testing)
        print("\nProcessing first 10 frames...")
        manifest_path = processor.process(
            show_progress=True,
            start_frame=0,
            end_frame=9,
        )

        print(f"\nProcessing complete!")
        print(f"Results saved to: {manifest_path}")

        return True

    except Exception as e:
        print(f"\nError processing video: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Test model loading
    success = test_tool_tracking_model()

    if success:
        print("\n" + "=" * 60)
        print("Basic test PASSED")
        print("=" * 60)

        # If a video path is provided, test with actual video
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            print(f"\nTesting with provided video: {video_path}")
            test_with_video(video_path)
        else:
            print("\nTo test with a video, run:")
            print("  python test_tool_tracking.py <path_to_surgical_video.mp4>")
    else:
        print("\n" + "=" * 60)
        print("Basic test FAILED")
        print("=" * 60)
        sys.exit(1)
