"""
ConvLSTM Surgical Tool Tracker model plugin for Lucidity.

This model detects and tracks surgical instruments in laparoscopic videos using
a weakly supervised deep learning approach combining ResNet and ConvLSTM.

Based on: Nwoye et al., 2019, IJCARS 14(6), pp.1059-1067
Repository: https://github.com/CAMMA-public/ConvLSTM-Surgical-Tool-Tracker
"""

import numpy as np
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from lucidity.base_model import BaseModel, ModelMetadata, ModelOutput, OutputType

# Add model directory to path for imports
model_dir = Path(__file__).parent


# Tool classes (from Cholec80 dataset)
TOOL_CLASSES = [
    'grasper',
    'bipolar',
    'hook',
    'scissors',
    'clipper',
    'irrigator',
    'specimen_bag'
]


class ToolTrackingModel(BaseModel):
    """
    ConvLSTM-based surgical tool detection and tracking model.

    This model processes laparoscopic video frames to detect and localise
    surgical instruments using a combination of ResNet feature extraction
    and ConvLSTM temporal modelling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the tool tracking model.

        Args:
            config: Optional configuration dictionary with keys:
                - checkpoint_path: Path to model checkpoint (default: ./checkpoints/ckpt)
                - num_classes: Number of tool classes (default: 7)
                - input_height: Input frame height (default: 480)
                - input_width: Input frame width (default: 854)
                - confidence_threshold: Minimum confidence for detections (default: 0.5)
        """
        super().__init__(config)
        self.config = config or {}

        # Model parameters
        self.checkpoint_path = self.config.get(
            'checkpoint_path',
            str(model_dir / 'checkpoints' / 'ckpt' / 'model.ckpt')
        )
        self.num_classes = self.config.get('num_classes', 7)
        self.input_height = self.config.get('input_height', 480)
        self.input_width = self.config.get('input_width', 854)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)

        # TensorFlow session and graph
        self.session = None
        self.graph = None
        self.img_placeholder = None
        self.seek_placeholder = None
        self.logits_tensor = None
        self.lhmaps_tensor = None

        # Frame tracking
        self.current_seek = 0

    def get_metadata(self) -> ModelMetadata:
        """Return metadata about this model."""
        return ModelMetadata(
            name="tool_tracking",
            version="1.0.0",
            description="ConvLSTM surgical tool detection and tracking for laparoscopic videos",
            author="CAMMA-public (adapted for Lucidity)",
            output_type=OutputType.BBOX,
            output_frequency="per_frame",
            dependencies=[
                "tensorflow>=2.10.0",
                "numpy>=1.21.0",
                "opencv-python>=4.0.0",
            ],
        )

    def initialize(self) -> None:
        """Initialise the model by loading weights and creating TensorFlow session."""
        try:
            import os

            # Suppress TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # Force TensorFlow 2.16+ to use legacy Keras 2
            os.environ['TF_USE_LEGACY_KERAS'] = '1'

            import tensorflow as tf

            print(f"  Loading ConvLSTM Tool Tracker from {self.checkpoint_path}")
            print(f"  TensorFlow version: {tf.__version__}")

            # Check TensorFlow version
            tf_version = int(tf.__version__.split('.')[0])

            if tf_version == 1:
                # TensorFlow 1.x compatibility mode
                self._initialize_tf1()
            else:
                # TensorFlow 2.x
                # Disable eager execution for TF1 compatibility
                tf.compat.v1.disable_eager_execution()
                self._initialize_tf1_compat()

            print(f"  Model loaded successfully ({self.num_classes} tool classes)")

        except Exception as e:
            raise RuntimeError(f"Failed to initialise tool tracking model: {e}")

    def _initialize_tf1(self) -> None:
        """Initialize with TensorFlow 1.x."""
        import tensorflow as tf
        import sys
        import os

        # Save current directory and change to model directory
        old_dir = os.getcwd()

        try:
            os.chdir(model_dir)
            sys.path.insert(0, str(model_dir))

            # Now import the model
            from model import Model as ConvLSTMModel
        finally:
            # Restore directory only (keep sys.path for model dependencies)
            os.chdir(old_dir)

        # Create new graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Create placeholders
            self.img_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None, 3],
                name='inputs'
            )
            self.seek_placeholder = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='seek'
            )

            # Expand dims and resize
            x = tf.expand_dims(self.img_placeholder, 0)
            x = tf.image.resize_bilinear(x, size=(self.input_height, self.input_width))

            # Build model
            network = ConvLSTMModel(
                images=x,
                seek=self.seek_placeholder,
                num_classes=self.num_classes
            )
            self.logits_tensor, self.lhmaps_tensor = network.build_model()

            # Create saver and session
            saver = tf.train.Saver()

            # Create session config for CPU
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
            self.session = tf.Session(config=config, graph=self.graph)

            # Restore weights
            saver.restore(self.session, self.checkpoint_path)

    def _initialize_tf1_compat(self) -> None:
        """Initialize with TensorFlow 2.x in TF1 compatibility mode."""
        import tensorflow as tf
        import sys
        import os

        # Save current directory and change to model directory
        old_dir = os.getcwd()

        try:
            os.chdir(model_dir)
            sys.path.insert(0, str(model_dir))

            print(f"  Current directory: {os.getcwd()}")
            print(f"  Sys.path[0]: {sys.path[0]}")
            print(f"  Looking for lib at: {model_dir / 'lib'}")
            print(f"  Lib exists: {(model_dir / 'lib').exists()}")

            # Now import the model
            from model import Model as ConvLSTMModel
        finally:
            # Restore directory only (keep sys.path for model dependencies)
            os.chdir(old_dir)

        # Create new graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            # Create placeholders using compat API
            self.img_placeholder = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=[None, None, 3],
                name='inputs'
            )
            self.seek_placeholder = tf.compat.v1.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='seek'
            )

            # Expand dims and resize
            x = tf.expand_dims(self.img_placeholder, 0)
            x = tf.image.resize(x, size=(self.input_height, self.input_width), method='bilinear')

            # Build model
            network = ConvLSTMModel(
                images=x,
                seek=self.seek_placeholder,
                num_classes=self.num_classes
            )
            self.logits_tensor, self.lhmaps_tensor = network.build_model()

            # Create saver and session
            saver = tf.compat.v1.train.Saver()

            # Create session config for CPU
            config = tf.compat.v1.ConfigProto(
                device_count={'GPU': 0},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
            self.session = tf.compat.v1.Session(config=config, graph=self.graph)

            # Restore weights
            saver.restore(self.session, self.checkpoint_path)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_number: int,
    ) -> Optional[ModelOutput]:
        """
        Process a single frame to detect and localise surgical tools.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
            timestamp: Timestamp in seconds from video start
            frame_number: Frame number in the video

        Returns:
            ModelOutput with tool detections and localisations
        """
        try:
            # Run inference
            logits, lhmaps = self.session.run(
                [self.logits_tensor, self.lhmaps_tensor],
                feed_dict={
                    self.img_placeholder: frame,
                    self.seek_placeholder: [self.current_seek]
                }
            )

            # Process outputs
            # logits: [1, num_classes] - tool presence probabilities
            # lhmaps: [1, 60, 107, num_classes] - localization heatmaps

            # Apply sigmoid to logits to get probabilities
            import tensorflow as tf
            probabilities = 1 / (1 + np.exp(-logits[0]))

            # Extract detections above threshold
            detections = []
            for tool_idx in range(self.num_classes):
                confidence = float(probabilities[tool_idx])

                if confidence >= self.confidence_threshold:
                    # Get localization heatmap for this tool
                    heatmap = lhmaps[0, :, :, tool_idx]

                    # Find centre of mass / max activation
                    coord = np.unravel_index(heatmap.argmax(), heatmap.shape)

                    # Scale coordinates to original frame size
                    # heatmap is 60x107, need to scale to frame size
                    cy = int((coord[0] / 60.0) * frame.shape[0])
                    cx = int((coord[1] / 107.0) * frame.shape[1])

                    # Extract bounding box from heatmap
                    # Threshold heatmap to get region
                    threshold = heatmap.max() * 0.3
                    binary_map = (heatmap > threshold).astype(np.uint8)

                    # Find bounding box of thresholded region
                    rows = np.any(binary_map, axis=1)
                    cols = np.any(binary_map, axis=0)

                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]

                        # Scale to original frame size
                        y1 = int((rmin / 60.0) * frame.shape[0])
                        y2 = int((rmax / 60.0) * frame.shape[0])
                        x1 = int((cmin / 107.0) * frame.shape[1])
                        x2 = int((cmax / 107.0) * frame.shape[1])
                    else:
                        # Use centre point with default size
                        box_size = 50
                        x1 = max(0, cx - box_size)
                        y1 = max(0, cy - box_size)
                        x2 = min(frame.shape[1], cx + box_size)
                        y2 = min(frame.shape[0], cy + box_size)

                    detections.append({
                        'tool': TOOL_CLASSES[tool_idx],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'centre': [cx, cy],
                        'heatmap': heatmap.tolist(),
                    })

            # Increment seek counter for temporal tracking
            self.current_seek += 1

            return ModelOutput(
                timestamp=timestamp,
                frame_number=frame_number,
                data={
                    'detections': detections,
                    'num_tools': len(detections),
                },
                confidence=float(probabilities.max()) if len(detections) > 0 else 0.0,
                metadata={
                    'seek': self.current_seek,
                    'all_probabilities': {
                        TOOL_CLASSES[i]: float(probabilities[i])
                        for i in range(self.num_classes)
                    }
                },
            )

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return None

    def cleanup(self) -> None:
        """Cleanup resources after processing is complete."""
        if self.session is not None:
            self.session.close()
            self.session = None

        self.graph = None
        print("  Tool tracking model cleaned up")


# Entry point for plugin discovery
def get_model_class():
    """Return the model class for plugin discovery."""
    return ToolTrackingModel
