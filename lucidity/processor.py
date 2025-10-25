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
from lucidity.masking import CircularMask, EndoscopicMaskDetector


class VideoProcessor:
    """
    Main processor for running models on videos and packaging outputs.
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        plugin_manager: Optional[PluginManager] = None,
        enable_masking: bool = False,
        masking_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video
            output_dir: Directory for outputs
            plugin_manager: Optional plugin manager (creates new one if None)
            enable_masking: Whether to enable endoscopic masking
            masking_config: Optional masking configuration
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

        # Masking configuration
        self.enable_masking = enable_masking
        self.masking_config = masking_config or {}
        self.mask: Optional[CircularMask] = None
        self.frames_for_mask: List[np.ndarray] = []

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
        output_format: Optional[str] = None,
        output_fps: Optional[float] = None,
    ) -> Path:
        """
        Process the video with all added models.

        Args:
            show_progress: Whether to show progress bar
            start_frame: First frame to process (0-indexed, inclusive). If None, starts from beginning.
            end_frame: Last frame to process (0-indexed, inclusive). If None, processes to end.
            output_format: Output format for frame-based models ('frames' or 'video'). If None, uses default.
            output_fps: FPS for video output. If None, uses input video FPS.

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
            # Apply masking if enabled
            if self.enable_masking:
                frame = self._apply_masking(frame, frame_number)
                if frame is None:
                    # Still collecting frames for mask detection
                    continue

            # Process with each model
            for model_name, model in self.models.items():
                if model.should_process_frame(timestamp, frame_number):
                    output = model.process_frame(frame, timestamp, frame_number)

                    if output is not None:
                        # Add to timeline
                        self.timeline.add_output(model_name, output)

        # Save outputs for each model
        print("Saving outputs...")

        # Determine output FPS
        video_fps = output_fps if output_fps is not None else self.video_reader.metadata.fps

        for model_name, model in self.models.items():
            print(f"  - {model_name}")
            output_files = self._save_model_outputs(
                model_name,
                model,
                model_output_dirs[model_name],
                output_format=output_format,
                output_fps=video_fps,
                apply_mask=self.enable_masking,
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

        # Save partial video if frame range was specified
        if start_frame is not None or end_frame is not None:
            print("Saving partial video...")
            partial_video_info = self._save_partial_video(
                start_frame=actual_start,
                end_frame=actual_end,
                apply_mask=self.enable_masking,
            )
            if partial_video_info:
                # Add to manifest as a special model entry
                from lucidity.base_model import ModelMetadata, OutputType
                manifest_builder.add_model(
                    model_metadata=ModelMetadata(
                        name="partial_video",
                        version="1.0.0",
                        description="Partial input video extracted from frame range",
                        output_type=OutputType.CUSTOM,
                        output_frequency="video",
                    ),
                    output_files=[partial_video_info],
                    total_outputs=1,
                    first_timestamp=actual_start / self.video_reader.metadata.fps,
                    last_timestamp=actual_end / self.video_reader.metadata.fps,
                )

        # Add masking information to manifest and save mask visualisation
        if self.enable_masking and self.mask is not None:
            manifest_builder.set_processing_info("masking", {
                "enabled": True,
                "centre": self.mask.centre,
                "radius": float(self.mask.radius),
                "config": self.masking_config,
            })

            # Save mask visualisation
            self._save_mask_visualisation()

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
        output_format: Optional[str] = None,
        output_fps: Optional[float] = None,
        apply_mask: bool = False,
    ) -> List[OutputFileInfo]:
        """
        Save outputs from a model to disk.

        Args:
            model_name: Name of the model
            model: Model instance
            output_dir: Directory to save outputs
            output_format: Output format for frame-based models ('frames' or 'video')
            output_fps: FPS for video output
            apply_mask: Whether to apply endoscopic mask to frame outputs

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
            # Determine output format
            format_choice = output_format if output_format else 'frames'

            if format_choice == 'video':
                # Save as video file
                video_path = output_dir / "output.mp4"
                self._save_frames_as_video(
                    entries=entries,
                    output_path=video_path,
                    fps=output_fps or 30.0,
                    apply_mask=apply_mask,
                )

                output_files.append(OutputFileInfo(
                    path=str(video_path.relative_to(self.output_dir)),
                    type="video",
                    format="mp4",
                    description=f"Video output ({len(entries)} frames @ {output_fps or 30.0} fps)",
                    size_bytes=video_path.stat().st_size,
                ))
            else:
                # Save as individual frame files (default)
                frames_dir = output_dir / "frames"
                frames_dir.mkdir(exist_ok=True)

                for entry in entries:
                    frame_data = entry.output.data

                    # Apply mask if enabled
                    if apply_mask and self.mask is not None:
                        frame_data = self.mask.apply(frame_data)

                    frame_path = frames_dir / f"frame_{entry.frame_number:06d}.npy"
                    np.save(frame_path, frame_data)

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
            json.dump(metadata.model_dump(mode='json'), f, indent=2)

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

    def _save_frames_as_video(
        self,
        entries: List,
        output_path: Path,
        fps: float,
        apply_mask: bool = False,
    ) -> None:
        """
        Save frame outputs as a video file.

        Args:
            entries: Timeline entries containing frame data
            output_path: Path to save the video
            fps: Frames per second for the video
            apply_mask: Whether to apply endoscopic mask to frames
        """
        import cv2

        if not entries:
            return

        # Get first frame to determine dimensions
        first_frame = entries[0].output.data

        # Check if we have a visualisation in metadata (for depth maps)
        if 'visualisation' in entries[0].output.metadata:
            # Use the pre-generated visualisation
            height, width = entries[0].output.metadata['visualisation'].shape[:2]
            use_visualisation = True
        else:
            # Process raw frame data
            if len(first_frame.shape) == 2:
                # Single channel (e.g., depth map) - needs normalization
                height, width = first_frame.shape
                use_visualisation = False
            elif len(first_frame.shape) == 3:
                # Multi-channel (e.g., RGB)
                height, width = first_frame.shape[:2]
                use_visualisation = False
            else:
                raise ValueError(f"Unsupported frame shape: {first_frame.shape}")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for entry in entries:
                frame_data = entry.output.data

                # Apply mask if enabled (before any processing)
                if apply_mask and self.mask is not None:
                    frame_data = self.mask.apply(frame_data)

                # Get frame to write
                if use_visualisation and 'visualisation' in entry.output.metadata:
                    # Use pre-generated colourmap visualisation
                    frame = entry.output.metadata['visualisation']

                    # Apply mask to visualisation if enabled
                    if apply_mask and self.mask is not None:
                        frame = self.mask.apply(frame)
                elif len(frame_data.shape) == 2:
                    # Single channel - normalise and apply colourmap
                    # Normalise to 0-255
                    frame_min = frame_data.min()
                    frame_max = frame_data.max()

                    if frame_max > frame_min:
                        normalised = ((frame_data - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        normalised = np.zeros_like(frame_data, dtype=np.uint8)

                    # Apply colourmap
                    frame = cv2.applyColorMap(normalised, cv2.COLORMAP_INFERNO)
                elif len(frame_data.shape) == 3:
                    # Multi-channel - assume RGB, convert to BGR for OpenCV
                    if frame_data.shape[2] == 3:
                        # Normalise if needed
                        if frame_data.max() <= 1.0:
                            frame = (frame_data * 255).astype(np.uint8)
                        else:
                            frame = frame_data.astype(np.uint8)

                        # Convert RGB to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        raise ValueError(f"Unsupported number of channels: {frame_data.shape[2]}")
                else:
                    raise ValueError(f"Unsupported frame shape: {frame_data.shape}")

                # Ensure frame is the right size
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))

                out.write(frame)

        finally:
            out.release()

        print(f"    Video saved to: {output_path.name}")

    def _apply_masking(self, frame: np.ndarray, frame_number: int) -> Optional[np.ndarray]:
        """
        Apply endoscopic masking to a frame.

        Collects initial frames for mask detection, then applies the detected mask.

        Args:
            frame: Input frame
            frame_number: Frame number

        Returns:
            Masked frame, or None if still collecting frames for mask detection
        """
        if self.mask is None:
            # Collect frames for mask detection
            n_frames = self.masking_config.get('n_frames', 10)

            if len(self.frames_for_mask) < n_frames:
                self.frames_for_mask.append(frame.copy())
                return None  # Don't process until we have the mask

            # Detect mask
            print(f"Detecting endoscopic mask from {len(self.frames_for_mask)} frames...")
            detector = EndoscopicMaskDetector(
                n_frames=len(self.frames_for_mask),
                black_threshold=self.masking_config.get('black_threshold', 30),
                min_valid_ratio=self.masking_config.get('min_valid_ratio', 0.3),
                morph_kernel_size=self.masking_config.get('morph_kernel_size', 5),
                circle_fit_method=self.masking_config.get('circle_fit_method', 'hough'),
            )

            frames_array = np.array(self.frames_for_mask)
            self.mask = detector.detect_mask_from_frames(frames_array)

            print(f"Mask detected: centre={self.mask.centre}, radius={self.mask.radius:.1f}")

            # Clear frames from memory
            self.frames_for_mask = []

        # Apply mask
        return self.mask.apply(frame)

    def _save_mask_visualisation(self) -> None:
        """Save a visualisation of the detected mask."""
        if self.mask is None:
            return

        import cv2

        # Create visualisation with mask overlay
        h, w = self.mask.mask.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw mask as white circle on black background
        mask_vis = np.stack([self.mask.mask * 255] * 3, axis=-1)
        vis = mask_vis

        # Draw circle outline and centre point
        centre_x, centre_y = self.mask.centre
        radius = int(self.mask.radius)

        cv2.circle(vis, (centre_x, centre_y), radius, (0, 255, 0), 2)
        cv2.circle(vis, (centre_x, centre_y), 5, (0, 255, 0), -1)

        # Add text with mask info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Centre: {self.mask.centre}, Radius: {self.mask.radius:.1f}"
        cv2.putText(vis, text, (10, 30), font, 0.7, (0, 255, 0), 2)

        # Save
        mask_path = self.output_dir / "mask_visualisation.png"
        cv2.imwrite(str(mask_path), vis)
        print(f"Mask visualisation saved to: {mask_path.name}")

    def _save_partial_video(
        self,
        start_frame: int,
        end_frame: int,
        apply_mask: bool = False,
    ) -> Optional[OutputFileInfo]:
        """
        Save a partial video from the original input when processing a frame range.

        Args:
            start_frame: First frame to include (0-indexed, inclusive)
            end_frame: Last frame to include (0-indexed, inclusive)
            apply_mask: Whether to apply endoscopic mask to the frames

        Returns:
            OutputFileInfo for the saved partial video, or None if failed
        """
        import cv2

        output_path = self.output_dir / "partial_video.mp4"

        # Get video metadata
        width = self.video_reader.metadata.width
        height = self.video_reader.metadata.height
        fps = self.video_reader.metadata.fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            # Read and write frames in the specified range
            for frame, timestamp, frame_number in self.video_reader.frames(start_frame=start_frame, end_frame=end_frame):
                # Apply mask if enabled
                if apply_mask and self.mask is not None:
                    frame = self.mask.apply(frame)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                out.write(frame_bgr)

        finally:
            out.release()

        # Reset the video reader position
        self.video_reader.reset()

        total_frames = end_frame - start_frame + 1
        duration = total_frames / fps

        print(f"Partial video saved: {output_path.name} ({total_frames} frames, {duration:.2f}s)")

        return OutputFileInfo(
            path=str(output_path.relative_to(self.output_dir)),
            type="video",
            format="mp4",
            description=f"Partial input video (frames {start_frame}-{end_frame}, {total_frames} frames)",
            size_bytes=output_path.stat().st_size,
        )
