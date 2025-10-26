"""
Efficient storage utilities for optical flow data.

This module provides utilities for saving and loading optical flow fields
in various formats with compression support.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import struct


def save_flow_flo(flow: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save optical flow in .flo format (Middlebury format).

    Args:
        flow: Flow field (H, W, 2) with (u, v) components
        path: Output path for .flo file
    """
    path = Path(path)
    h, w = flow.shape[:2]

    with open(path, 'wb') as f:
        # Write header
        f.write(b'PIEH')  # Magic number
        f.write(struct.pack('i', w))
        f.write(struct.pack('i', h))

        # Write flow data (interleaved u, v)
        flow_data = flow.astype(np.float32).reshape(-1)
        flow_data.tofile(f)


def load_flow_flo(path: Union[str, Path]) -> np.ndarray:
    """
    Load optical flow from .flo format (Middlebury format).

    Args:
        path: Path to .flo file

    Returns:
        Flow field (H, W, 2) with (u, v) components
    """
    path = Path(path)

    with open(path, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'PIEH':
            raise ValueError(f"Invalid .flo file: {path}")

        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]

        # Read flow data
        flow_data = np.fromfile(f, dtype=np.float32)
        flow = flow_data.reshape(h, w, 2)

    return flow


def save_flow_compressed(
    flow: np.ndarray,
    path: Union[str, Path],
    compression_level: int = 6,
    save_metadata: bool = True,
    timestamp: Optional[float] = None,
    frame_number: Optional[int] = None,
) -> int:
    """
    Save optical flow with compression in .npz format.

    This provides efficient storage with typical compression ratios of 10-20x
    for optical flow data.

    Args:
        flow: Flow field (H, W, 2) with (u, v) components
        path: Output path for .npz file
        compression_level: Compression level (0-9, higher = more compression)
        save_metadata: Whether to include metadata
        timestamp: Optional timestamp
        frame_number: Optional frame number

    Returns:
        File size in bytes
    """
    path = Path(path)

    # Prepare data dictionary
    save_dict = {'flow': flow.astype(np.float32)}

    if save_metadata:
        # Add metadata
        save_dict['shape'] = np.array(flow.shape, dtype=np.int32)
        save_dict['dtype'] = np.array([str(flow.dtype)])

        if timestamp is not None:
            save_dict['timestamp'] = np.array([timestamp], dtype=np.float64)
        if frame_number is not None:
            save_dict['frame_number'] = np.array([frame_number], dtype=np.int32)

    # Save with compression
    np.savez_compressed(path, **save_dict)

    return path.stat().st_size


def load_flow_compressed(path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Load optical flow from compressed .npz format.

    Args:
        path: Path to .npz file

    Returns:
        Tuple of (flow field, metadata dict)
    """
    path = Path(path)

    with np.load(path) as data:
        flow = data['flow']

        metadata = {}
        if 'timestamp' in data:
            metadata['timestamp'] = float(data['timestamp'][0])
        if 'frame_number' in data:
            metadata['frame_number'] = int(data['frame_number'][0])

    return flow, metadata


def save_sparse_flow_sequence(
    sparse_flows: list,
    timestamps: np.ndarray,
    frame_numbers: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[dict] = None,
) -> int:
    """
    Save a sequence of sparse optical flow vectors in a single compressed file.

    This is extremely efficient for sparse flow storage as it only saves
    the non-zero vectors with their coordinates.

    Args:
        sparse_flows: List of sparse flow dicts with 'x', 'y', 'u', 'v'
        timestamps: Timestamps for each flow (N,)
        frame_numbers: Frame numbers for each flow (N,)
        path: Output path for .npz file
        metadata: Optional metadata dictionary

    Returns:
        File size in bytes
    """
    path = Path(path)

    # Combine all sparse flows with frame indices
    all_x, all_y, all_u, all_v, all_frame_idx = [], [], [], [], []

    for i, flow in enumerate(sparse_flows):
        n_vectors = len(flow['x'])
        if n_vectors > 0:
            all_x.append(flow['x'])
            all_y.append(flow['y'])
            all_u.append(flow['u'])
            all_v.append(flow['v'])
            all_frame_idx.append(np.full(n_vectors, i, dtype=np.int32))

    # Concatenate all arrays
    if len(all_x) > 0:
        x_combined = np.concatenate(all_x)
        y_combined = np.concatenate(all_y)
        u_combined = np.concatenate(all_u)
        v_combined = np.concatenate(all_v)
        frame_idx_combined = np.concatenate(all_frame_idx)
    else:
        # No vectors at all
        x_combined = np.array([], dtype=np.float32)
        y_combined = np.array([], dtype=np.float32)
        u_combined = np.array([], dtype=np.float32)
        v_combined = np.array([], dtype=np.float32)
        frame_idx_combined = np.array([], dtype=np.int32)

    # Build save dictionary
    save_dict = {
        'x': x_combined,
        'y': y_combined,
        'u': u_combined,
        'v': v_combined,
        'frame_idx': frame_idx_combined,
        'timestamps': timestamps.astype(np.float64),
        'frame_numbers': frame_numbers.astype(np.int32),
    }

    # Add metadata if provided
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                save_dict[key] = np.array([value])
            elif isinstance(value, (list, tuple)):
                save_dict[key] = np.array(value)

    # Save with compression
    np.savez_compressed(path, **save_dict)

    return path.stat().st_size


def load_sparse_flow_sequence(path: Union[str, Path]) -> Tuple[list, np.ndarray, np.ndarray, dict]:
    """
    Load a sequence of sparse optical flow vectors from compressed file.

    Args:
        path: Path to .npz file

    Returns:
        Tuple of (sparse_flows, timestamps, frame_numbers, metadata)
        - sparse_flows: List of dicts with 'x', 'y', 'u', 'v'
        - timestamps: (N,) array
        - frame_numbers: (N,) array
        - metadata: Dictionary with additional metadata
    """
    path = Path(path)

    with np.load(path) as data:
        x_all = data['x']
        y_all = data['y']
        u_all = data['u']
        v_all = data['v']
        frame_idx = data['frame_idx']
        timestamps = data['timestamps']
        frame_numbers = data['frame_numbers']

        # Reconstruct sparse flows for each frame
        n_frames = len(timestamps)
        sparse_flows = []

        for i in range(n_frames):
            mask = frame_idx == i
            sparse_flows.append({
                'x': x_all[mask],
                'y': y_all[mask],
                'u': u_all[mask],
                'v': v_all[mask],
            })

        # Extract metadata
        metadata = {}
        for key in data.files:
            if key not in ['x', 'y', 'u', 'v', 'frame_idx', 'timestamps', 'frame_numbers']:
                metadata[key] = data[key]

    return sparse_flows, timestamps, frame_numbers, metadata


def save_flow_sequence_compressed(
    flows: list,
    timestamps: np.ndarray,
    frame_numbers: np.ndarray,
    path: Union[str, Path],
    compression_level: int = 6,
) -> int:
    """
    Save a sequence of dense optical flow fields in a single compressed file.

    This is the most efficient storage method for long sequences as it:
    1. Avoids file system overhead of many small files
    2. Allows better compression across the sequence
    3. Enables fast batch loading

    Args:
        flows: List of flow fields (H, W, 2)
        timestamps: Timestamps for each flow (N,)
        frame_numbers: Frame numbers for each flow (N,)
        path: Output path for .npz file
        compression_level: Compression level (0-9)

    Returns:
        File size in bytes
    """
    path = Path(path)

    # Stack flows
    flows_stacked = np.stack(flows, axis=0)  # (N, H, W, 2)

    # Save with compression
    np.savez_compressed(
        path,
        flows=flows_stacked.astype(np.float32),
        timestamps=timestamps.astype(np.float64),
        frame_numbers=frame_numbers.astype(np.int32),
    )

    return path.stat().st_size


def load_flow_sequence_compressed(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a sequence of optical flow fields from compressed file.

    Args:
        path: Path to .npz file

    Returns:
        Tuple of (flows, timestamps, frame_numbers)
        - flows: (N, H, W, 2) array
        - timestamps: (N,) array
        - frame_numbers: (N,) array
    """
    path = Path(path)

    with np.load(path) as data:
        flows = data['flows']
        timestamps = data['timestamps']
        frame_numbers = data['frame_numbers']

    return flows, timestamps, frame_numbers


def estimate_compression_ratio(flow: np.ndarray, compression_level: int = 6) -> float:
    """
    Estimate the compression ratio for a flow field.

    Args:
        flow: Flow field (H, W, 2)
        compression_level: Compression level (0-9)

    Returns:
        Estimated compression ratio (original_size / compressed_size)
    """
    # Calculate uncompressed size
    uncompressed_size = flow.nbytes

    # Create temporary compressed version
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, flow=flow.astype(np.float32))
    compressed_size = len(buffer.getvalue())

    return uncompressed_size / compressed_size if compressed_size > 0 else 1.0
