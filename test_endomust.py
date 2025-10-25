"""
Simple test script for the EndoMUST depth model plugin.
"""

import sys
import os
import numpy as np
import cv2

# Add the models directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from endomust_depth import EndoMUSTDepthModel


def test_endomust_model():
    """Test the EndoMUST depth model with a dummy frame."""
    print("Testing EndoMUST depth model...")
    print("-" * 60)

    # Create model instance
    config = {
        'backbone_size': 'base',
        'lora_rank': 4,
        'lora_type': 'dvlora',
        'image_shape': (224, 280),
        'min_depth': 0.001,
        'max_depth': 150.0,
    }

    model = EndoMUSTDepthModel(config)

    # Print metadata
    metadata = model.get_metadata()
    print(f"\nModel Metadata:")
    print(f"  Name: {metadata.name}")
    print(f"  Version: {metadata.version}")
    print(f"  Description: {metadata.description}")
    print(f"  Output Type: {metadata.output_type}")
    print(f"  Dependencies: {', '.join(metadata.dependencies)}")

    # Initialize the model
    print(f"\n{'-' * 60}")
    model.initialize()

    # Create a dummy frame (640x480 RGB)
    print(f"\n{'-' * 60}")
    print("Processing test frame...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Process the frame
    output = model.process_frame(
        frame=dummy_frame,
        timestamp=0.0,
        frame_number=0
    )

    print(f"\nOutput:")
    print(f"  Timestamp: {output.timestamp}s")
    print(f"  Frame number: {output.frame_number}")
    print(f"  Depth map shape: {output.data.shape}")
    print(f"  Depth range: [{output.data.min():.3f}, {output.data.max():.3f}]")
    print(f"  Depth mean: {output.data.mean():.3f}")
    print(f"  Metadata keys: {list(output.metadata.keys())}")

    # Save visualisation if available
    if 'visualisation' in output.metadata:
        vis = output.metadata['visualisation']
        output_path = 'test_depth_output.png'
        cv2.imwrite(output_path, vis)
        print(f"\nVisualisation saved to: {output_path}")

    # Cleanup
    model.cleanup()

    print(f"\n{'-' * 60}")
    print("Test completed successfully!")
    print("-" * 60)


if __name__ == "__main__":
    test_endomust_model()
