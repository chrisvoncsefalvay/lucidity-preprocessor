"""Command-line interface for Lucidity Preprocessor."""

import click
from pathlib import Path
from typing import List, Optional
import json

from lucidity.plugin_manager import get_plugin_manager
from lucidity.processor import VideoProcessor
from lucidity.manifest import load_manifest, create_output_package_summary


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Lucidity Preprocessor - Modular video ML inference pipeline."""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--models', '-m',
    required=True,
    help='Comma-separated list of models to use (e.g., "pose_detection,scene_segmentation")',
)
@click.option(
    '--output', '-o',
    default='./output',
    type=click.Path(),
    help='Output directory for results (default: ./output)',
)
@click.option(
    '--model-config',
    type=click.Path(exists=True),
    help='JSON file with model configurations',
)
@click.option(
    '--discover-dir',
    type=click.Path(exists=True),
    help='Additional directory to discover plugins from',
)
@click.option(
    '--no-progress',
    is_flag=True,
    help='Disable progress bar',
)
@click.option(
    '--start-time',
    type=float,
    help='Start time in seconds (e.g., 10.5 for 10.5 seconds)',
)
@click.option(
    '--end-time',
    type=float,
    help='End time in seconds (e.g., 60.0 for 1 minute)',
)
@click.option(
    '--start-frame',
    type=int,
    help='Start frame number (0-indexed)',
)
@click.option(
    '--end-frame',
    type=int,
    help='End frame number (0-indexed, inclusive)',
)
@click.option(
    '--output-format',
    type=click.Choice(['frames', 'video'], case_sensitive=False),
    help='Output format for frame-based models: "frames" (individual files) or "video" (MP4)',
)
@click.option(
    '--fps',
    type=float,
    help='FPS for video output (defaults to input video FPS)',
)
@click.option(
    '--mask',
    is_flag=True,
    help='Enable endoscopic masking (detects and masks circular region)',
)
@click.option(
    '--mask-frames',
    type=int,
    default=10,
    help='Number of frames to analyse for mask detection (default: 10)',
)
@click.option(
    '--mask-threshold',
    type=int,
    default=30,
    help='Black pixel threshold for masking (0-255, default: 30)',
)
@click.option(
    '--mask-method',
    type=click.Choice(['hough', 'contour'], case_sensitive=False),
    default='hough',
    help='Circle fitting method: hough or contour (default: hough)',
)
def process(
    video_path: str,
    models: str,
    output: str,
    model_config: Optional[str],
    discover_dir: Optional[str],
    no_progress: bool,
    start_time: Optional[float],
    end_time: Optional[float],
    start_frame: Optional[int],
    end_frame: Optional[int],
    output_format: Optional[str],
    fps: Optional[float],
    mask: bool,
    mask_frames: int,
    mask_threshold: int,
    mask_method: str,
):
    """
    Process a video with selected models.

    Example:
        lucidity process video.mp4 --models pose_detection,scene_segmentation

        # Process only frames 100-500
        lucidity process video.mp4 --models my_model --start-frame 100 --end-frame 500

        # Process only 10-30 seconds of video
        lucidity process video.mp4 --models my_model --start-time 10 --end-time 30

        # Process with endoscopic masking
        lucidity process video.mp4 --models my_model --mask

        # Process with custom masking parameters
        lucidity process video.mp4 --models my_model --mask --mask-frames 20 --mask-threshold 40
    """
    # Validate that frame and time options are not mixed
    if (start_time is not None or end_time is not None) and (start_frame is not None or end_frame is not None):
        click.echo("Error: Cannot mix --start-time/--end-time with --start-frame/--end-frame", err=True)
        click.echo("Please use either time-based or frame-based options, not both.", err=True)
        return

    # Get plugin manager and discover plugins
    plugin_manager = get_plugin_manager()

    if discover_dir:
        plugin_manager.discover_from_directory(Path(discover_dir))

    # Parse model list
    model_names = [m.strip() for m in models.split(',')]

    # Load model configs if provided
    model_configs = {}
    if model_config:
        with open(model_config, 'r') as f:
            model_configs = json.load(f)

    # Validate models exist
    available_models = plugin_manager.list_plugins()
    for model_name in model_names:
        if model_name not in available_models:
            click.echo(f"Error: Model '{model_name}' not found.", err=True)
            click.echo(f"Available models: {', '.join(available_models)}", err=True)
            return

    # Create processor
    click.echo(f"Processing video: {video_path}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Models: {', '.join(model_names)}")

    # Convert time ranges to frame ranges if needed
    frame_range_start = start_frame
    frame_range_end = end_frame

    if start_time is not None or end_time is not None:
        # Need to read video metadata to convert timestamps
        from lucidity.video import VideoReader
        reader = VideoReader(video_path)

        if start_time is not None:
            frame_range_start = int(start_time * reader.metadata.fps)
            click.echo(f"Start time: {start_time}s (frame {frame_range_start})")

        if end_time is not None:
            frame_range_end = int(end_time * reader.metadata.fps)
            click.echo(f"End time: {end_time}s (frame {frame_range_end})")

        reader.close()
    elif start_frame is not None or end_frame is not None:
        if start_frame is not None:
            click.echo(f"Start frame: {start_frame}")
        if end_frame is not None:
            click.echo(f"End frame: {end_frame}")

    click.echo()

    # Configure masking if enabled
    masking_config = None
    if mask:
        click.echo("Endoscopic masking: enabled")
        click.echo(f"  Mask detection frames: {mask_frames}")
        click.echo(f"  Black threshold: {mask_threshold}")
        click.echo(f"  Circle fitting method: {mask_method}")
        click.echo()

        masking_config = {
            'n_frames': mask_frames,
            'black_threshold': mask_threshold,
            'circle_fit_method': mask_method,
        }

    processor = VideoProcessor(
        video_path=video_path,
        output_dir=output,
        plugin_manager=plugin_manager,
        enable_masking=mask,
        masking_config=masking_config,
    )

    # Add models
    for model_name in model_names:
        config = model_configs.get(model_name)
        processor.add_model(model_name, config)

    # Process
    try:
        manifest_path = processor.process(
            show_progress=not no_progress,
            start_frame=frame_range_start,
            end_frame=frame_range_end,
            output_format=output_format,
            output_fps=fps,
        )
        click.echo(f"\nProcessing complete!")
        click.echo(f"Manifest saved to: {manifest_path}")
    except Exception as e:
        click.echo(f"Error during processing: {e}", err=True)
        raise


@cli.command()
@click.option(
    '--discover-dir',
    type=click.Path(exists=True),
    help='Additional directory to discover plugins from',
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed information about each model',
)
def list_models(discover_dir: Optional[str], verbose: bool):
    """List all available model plugins."""
    plugin_manager = get_plugin_manager()

    if discover_dir:
        plugin_manager.discover_from_directory(Path(discover_dir))

    models = plugin_manager.list_plugins()

    if not models:
        click.echo("No models found.")
        click.echo("\nTo add models:")
        click.echo("  1. Install model plugins via pip")
        click.echo("  2. Use --discover-dir to scan a directory")
        click.echo("  3. Place models in ./models/ directory")
        return

    click.echo(f"Found {len(models)} model(s):\n")

    if verbose:
        metadata_dict = plugin_manager.get_all_metadata()
        for name in sorted(models):
            metadata = metadata_dict.get(name)
            if metadata:
                click.echo(f"  {name}")
                click.echo(f"    Version: {metadata.version}")
                click.echo(f"    Description: {metadata.description}")
                click.echo(f"    Output Type: {metadata.output_type}")
                click.echo(f"    Frequency: {metadata.output_frequency}")
                if metadata.author:
                    click.echo(f"    Author: {metadata.author}")
                click.echo()
    else:
        for name in sorted(models):
            click.echo(f"  - {name}")


@cli.command()
@click.argument('model_name')
@click.option(
    '--discover-dir',
    type=click.Path(exists=True),
    help='Additional directory to discover plugins from',
)
def model_info(model_name: str, discover_dir: Optional[str]):
    """Show detailed information about a specific model."""
    plugin_manager = get_plugin_manager()

    if discover_dir:
        plugin_manager.discover_from_directory(Path(discover_dir))

    try:
        metadata = plugin_manager.get_plugin_metadata(model_name)

        click.echo(f"Model: {metadata.name}")
        click.echo(f"Version: {metadata.version}")
        click.echo(f"Description: {metadata.description}")
        click.echo(f"Output Type: {metadata.output_type}")
        click.echo(f"Output Frequency: {metadata.output_frequency}")

        if metadata.frame_rate:
            click.echo(f"Frame Rate: {metadata.frame_rate} Hz")

        if metadata.author:
            click.echo(f"Author: {metadata.author}")

        if metadata.dependencies:
            click.echo(f"\nDependencies:")
            for dep in metadata.dependencies:
                click.echo(f"  - {dep}")

        # Get output schema if available
        model = plugin_manager.get_plugin(model_name)
        schema = model.get_output_schema()
        if schema.get('description'):
            click.echo(f"\nOutput Schema:")
            click.echo(f"  {schema['description']}")

    except KeyError:
        click.echo(f"Error: Model '{model_name}' not found.", err=True)
        click.echo(f"\nUse 'lucidity list-models' to see available models.", err=True)


@cli.command()
@click.argument('manifest_path', type=click.Path(exists=True))
def show_manifest(manifest_path: str):
    """Display information from a processing manifest."""
    try:
        manifest = load_manifest(Path(manifest_path))
        summary = create_output_package_summary(manifest)
        click.echo(summary)
    except Exception as e:
        click.echo(f"Error reading manifest: {e}", err=True)


@cli.command()
@click.argument('output_name')
@click.option(
    '--type', '-t',
    'output_type',
    type=click.Choice(['frame', 'keypoints', 'bbox', 'label', 'embedding', 'custom']),
    default='custom',
    help='Type of output the model produces',
)
@click.option(
    '--frequency', '-f',
    default='per_frame',
    help='Output frequency (e.g., "per_frame", "1.0" for 1 Hz)',
)
def create_template(output_name: str, output_type: str, frequency: str):
    """
    Create a template for a new model plugin.

    Example:
        lucidity create-template my_model --type keypoints --frequency per_frame
    """
    template = f'''"""
{output_name} model plugin for Lucidity.

This is a template - implement the TODOs below.
"""

import numpy as np
from typing import Optional, Dict, Any

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType


class {output_name.title().replace('_', '')}Model(BaseModel):
    """
    TODO: Add description of what this model does.
    """

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="{output_name}",
            version="0.1.0",
            description="TODO: Add description",
            author="TODO: Add your name",
            output_type=OutputType.{output_type.upper()},
            output_frequency="{frequency}",
            dependencies=[
                # TODO: List required packages
            ],
        )

    def initialize(self) -> None:
        """Initialize the model (load weights, setup, etc.)."""
        # TODO: Load model, initialize resources
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput if the model produces output for this frame, None otherwise
        """
        # TODO: Implement your model inference here

        # Example output structure:
        data = {{'example': 'data'}}  # Replace with actual output

        return ModelOutput(
            timestamp=timestamp,
            frame_number=frame_number,
            data=data,
            confidence=None,  # Optional confidence score
            metadata={{}},  # Optional additional metadata
        )

    def cleanup(self) -> None:
        """Cleanup resources after processing is complete."""
        # TODO: Release resources, close files, etc.
        pass


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return {output_name.title().replace('_', '')}Model
'''

    output_path = Path(f"{output_name}_model.py")
    with open(output_path, 'w') as f:
        f.write(template)

    click.echo(f"Template created: {output_path}")
    click.echo("\nNext steps:")
    click.echo("  1. Edit the template and implement the TODOs")
    click.echo("  2. Place the file in a models/ directory or install as a package")
    click.echo("  3. Use --discover-dir to load it, or register it as an entry point")


if __name__ == '__main__':
    cli()
