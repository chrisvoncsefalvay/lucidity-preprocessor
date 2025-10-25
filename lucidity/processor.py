"""Video processor that orchestrates model execution and output packaging."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from lucidity.video import VideoReader
from lucidity.timeline import Timeline
from lucidity.manifest import ManifestBuilder, OutputFileInfo, create_output_package_summary
from lucidity.plugin_manager import PluginManager
from lucidity.base_model import BaseModel, OutputType


class VideoProcessor:
    """
    Main processor for running models on videos and packaging outputs.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        plugin_manager: Optional[PluginManager] = None,
    ):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video
            output_dir: Directory for outputs
            plugin_manager: Optional plugin manager (creates new one if None)
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plugin_manager = plugin_manager or PluginManager()
        self.video_reader = VideoReader(str(self.video_path))
        self.timeline = Timeline(
            self.video_reader.metadata.fps,
            self.video_reader.metadata.total_frames,
        )

        self.models: Dict[str, BaseModel] = {}

    def add_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a model to the processing pipeline.

        Args:
            model_name: Name of the model plugin
            config: Optional configuration for the model
        """
        model = self.plugin_manager.get_plugin(model_name, config)
        self.models[model_name] = model

    def process(
        self,
        show_progress: bool = True,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Path:
        """
        Process the video with all added models.

        Args:
            show_progress: Whether to show progress bar
            start_frame: First frame to process (0-indexed, inclusive). If None, starts from beginning.
            end_frame: Last frame to process (0-indexed, inclusive). If None, processes to end.

        Returns:
            Path to the manifest file
        """
        # Determine actual frame range
        actual_start = start_frame if start_frame is not None else 0
        actual_end = end_frame if end_frame is not None else self.video_reader.metadata.total_frames - 1

        # Validate and clamp range
        actual_start = max(0, actual_start)
        actual_end = min(actual_end, self.video_reader.metadata.total_frames - 1)

        total_frames_to_process = actual_end - actual_start + 1

        # Initialize manifest builder
        manifest_builder = ManifestBuilder(
            self.video_reader.metadata,
            self.output_dir,
        )
        manifest_builder.start_processing()

        # Add frame range information to manifest
        if start_frame is not None or end_frame is not None:
            manifest_builder.set_processing_info("frame_range", {
                "start_frame": actual_start,
                "end_frame": actual_end,
                "total_frames_processed": total_frames_to_process,
            })

        # Initialize all models
        print(f"Initialising {len(self.models)} models...")
        for name, model in self.models.items():
            print(f"  - {name}")
            model.initialize()

        # Create output directories for each model
        model_output_dirs = {}
        for name in self.models.keys():
            model_dir = self.output_dir / name
            model_dir.mkdir(exist_ok=True)
            model_output_dirs[name] = model_dir

        # Process video frames
        if start_frame is not None or end_frame is not None:
            print(f"Processing frames {actual_start} to {actual_end} ({total_frames_to_process} frames)...")
        else:
            print(f"Processing {self.video_reader.metadata.total_frames} frames...")

        frames_iterator = self.video_reader.frames(start_frame=start_frame, end_frame=end_frame)
        if show_progress:
            frames_iterator = tqdm(
                frames_iterator,
                total=total_frames_to_process,
                desc="Processing",
            )

        for frame, timestamp, frame_number in frames_iterator:
            # Process with each model
            for model_name, model in self.models.items():
                if model.should_process_frame(timestamp, frame_number):
                    output = model.process_frame(frame, timestamp, frame_number)

                    if output is not None:
                        # Add to timeline
                        self.timeline.add_output(model_name, output)

        # Save outputs for each model
        print("Saving outputs...")
        for model_name, model in self.models.items():
            print(f"  - {model_name}")
            output_files = self._save_model_outputs(
                model_name,
                model,
                model_output_dirs[model_name],
            )

            # Get output statistics
            entries = self.timeline.get_model_outputs(model_name)
            if entries:
                timestamps = [e.timestamp for e in entries]
                frame_numbers = [e.frame_number for e in entries if e.frame_number is not None]

                manifest_builder.add_model(
                    model_metadata=model.get_metadata(),
                    output_files=output_files,
                    total_outputs=len(entries),
                    first_timestamp=min(timestamps),
                    last_timestamp=max(timestamps),
                    frame_coverage={
                        "frame_numbers": frame_numbers,
                        "total_frames": len(frame_numbers),
                    },
                )

        # Cleanup models
        print("Cleaning up...")
        for model in self.models.values():
            model.cleanup()

        # Save timeline summary
        manifest_builder.set_timeline_summary(self.timeline.get_timeline_summary())

        # End processing and save manifest
        manifest_builder.end_processing()
        manifest_path = manifest_builder.save()

        # Print summary
        manifest = manifest_builder.build()
        print("\n" + create_output_package_summary(manifest))

        return manifest_path

    def _save_model_outputs(
        self,
        model_name: str,
        model: BaseModel,
        output_dir: Path,
    ) -> List[OutputFileInfo]:
        """
        Save outputs from a model to disk.

        Args:
            model_name: Name of the model
            model: Model instance
            output_dir: Directory to save outputs

        Returns:
            List of OutputFileInfo for saved files
        """
        output_files = []
        entries = self.timeline.get_model_outputs(model_name)

        if not entries:
            return output_files

        metadata = model.get_metadata()

        # Save outputs based on type
        if metadata.output_type == OutputType.FRAME:
            # Save as images or video
            frames_dir = output_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            for entry in entries:
                frame_path = frames_dir / f"frame_{entry.frame_number:06d}.npy"
                np.save(frame_path, entry.output.data)

            output_files.append(OutputFileInfo(
                path=str(frames_dir.relative_to(self.output_dir)),
                type="frames",
                format="npy",
                description=f"Frame outputs ({len(entries)} frames)",
            ))

        elif metadata.output_type in [OutputType.KEYPOINTS, OutputType.BBOX, OutputType.LABEL]:
            # Save as JSON
            data_file = output_dir / "outputs.json"
            outputs_data = []

            for entry in entries:
                output_dict = {
                    "timestamp": entry.timestamp,
                    "frame_number": entry.frame_number,
                    "data": self._serialize_data(entry.output.data),
                    "confidence": entry.output.confidence,
                    "metadata": entry.output.metadata,
                }
                outputs_data.append(output_dict)

            with open(data_file, 'w') as f:
                json.dump(outputs_data, f, indent=2)

            output_files.append(OutputFileInfo(
                path=str(data_file.relative_to(self.output_dir)),
                type="data",
                format="json",
                description=f"Structured outputs ({len(entries)} items)",
                size_bytes=data_file.stat().st_size,
            ))

        elif metadata.output_type in [OutputType.EMBEDDING, OutputType.TIMESERIES]:
            # Save as numpy array
            data_file = output_dir / "embeddings.npz"

            # Collect all data
            timestamps = np.array([e.timestamp for e in entries])
            frame_numbers = np.array([e.frame_number for e in entries])
            data_arrays = [e.output.data for e in entries]

            # Stack if possible
            try:
                stacked_data = np.stack(data_arrays)
                np.savez(
                    data_file,
                    data=stacked_data,
                    timestamps=timestamps,
                    frame_numbers=frame_numbers,
                )
            except:
                # Save as list if stacking fails
                np.savez(
                    data_file,
                    data=np.array(data_arrays, dtype=object),
                    timestamps=timestamps,
                    frame_numbers=frame_numbers,
                )

            output_files.append(OutputFileInfo(
                path=str(data_file.relative_to(self.output_dir)),
                type="data",
                format="npz",
                description=f"Embeddings/timeseries ({len(entries)} items)",
                size_bytes=data_file.stat().st_size,
            ))

        else:  # CUSTOM
            # Save as JSON
            data_file = output_dir / "outputs.json"
            outputs_data = []

            for entry in entries:
                output_dict = {
                    "timestamp": entry.timestamp,
                    "frame_number": entry.frame_number,
                    "data": self._serialize_data(entry.output.data),
                    "metadata": entry.output.metadata,
                }
                outputs_data.append(output_dict)

            with open(data_file, 'w') as f:
                json.dump(outputs_data, f, indent=2)

            output_files.append(OutputFileInfo(
                path=str(data_file.relative_to(self.output_dir)),
                type="data",
                format="json",
                description=f"Custom outputs ({len(entries)} items)",
                size_bytes=data_file.stat().st_size,
            ))

        # Save model metadata
        metadata_file = output_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(json.loads(metadata.json()), f, indent=2)

        output_files.append(OutputFileInfo(
            path=str(metadata_file.relative_to(self.output_dir)),
            type="metadata",
            format="json",
            description="Model metadata",
            size_bytes=metadata_file.stat().st_size,
        ))

        return output_files

    def _serialize_data(self, data: Any) -> Any:
        """
        Serialize data for JSON output.

        Args:
            data: Data to serialize

        Returns:
            JSON-serializable version of data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        else:
            return data
