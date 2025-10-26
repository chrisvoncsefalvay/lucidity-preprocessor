"""
Utilities for plotting sparse optical flow vector fields.

This module provides functions for visualising sparse optical flow vectors
from the RAFT model, including quiver plots and colour-coded visualisations.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_sparse_flow_quiver(
    sparse_flow: dict,
    image_shape: Tuple[int, int],
    background: Optional[np.ndarray] = None,
    scale: float = 1.0,
    color: str = 'cyan',
    alpha: float = 0.8,
    arrow_width: float = 0.003,
    figsize: Tuple[int, int] = (10, 10),
) -> Figure:
    """
    Plot sparse optical flow as quiver (arrow) plot.

    Args:
        sparse_flow: Dictionary with 'x', 'y', 'u', 'v' arrays
        image_shape: (height, width) of the image
        background: Optional background image (H, W, 3) to overlay arrows on
        scale: Scale factor for arrow lengths (higher = shorter arrows)
        color: Arrow colour
        alpha: Arrow transparency (0-1)
        arrow_width: Width of arrow shafts
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object
    """
    h, w = image_shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Show background if provided
    if background is not None:
        ax.imshow(background)
    else:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')

    # Plot flow vectors
    if len(sparse_flow['x']) > 0:
        ax.quiver(
            sparse_flow['x'],
            sparse_flow['y'],
            sparse_flow['u'],
            sparse_flow['v'],
            color=color,
            alpha=alpha,
            scale_units='xy',
            scale=scale,
            width=arrow_width,
            angles='xy',
        )

    ax.set_title(f'Sparse Optical Flow ({len(sparse_flow["x"])} vectors)')
    ax.axis('off')
    fig.tight_layout()

    return fig


def plot_sparse_flow_color(
    sparse_flow: dict,
    image_shape: Tuple[int, int],
    background: Optional[np.ndarray] = None,
    point_size: int = 20,
    figsize: Tuple[int, int] = (10, 10),
) -> Figure:
    """
    Plot sparse optical flow with colour-coded points.

    Colour represents flow direction (hue) and magnitude (saturation).

    Args:
        sparse_flow: Dictionary with 'x', 'y', 'u', 'v' arrays
        image_shape: (height, width) of the image
        background: Optional background image (H, W, 3) to overlay points on
        point_size: Size of scatter points
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object
    """
    h, w = image_shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Show background if provided
    if background is not None:
        ax.imshow(background)
    else:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')

    # Compute flow magnitude and angle
    if len(sparse_flow['x']) > 0:
        magnitudes = np.sqrt(sparse_flow['u']**2 + sparse_flow['v']**2)
        angles = np.arctan2(sparse_flow['v'], sparse_flow['u'])

        # Convert to HSV colour
        # Hue from angle, Value from magnitude
        hues = (angles + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        saturations = np.ones_like(hues)
        values = np.clip(magnitudes / magnitudes.max() if magnitudes.max() > 0 else 1, 0, 1)

        # Convert HSV to RGB
        hsv = np.stack([hues, saturations, values], axis=1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv.reshape(1, -1, 3)).reshape(-1, 3)

        # Plot as scatter
        ax.scatter(
            sparse_flow['x'],
            sparse_flow['y'],
            c=rgb,
            s=point_size,
            alpha=0.8,
        )

    ax.set_title(f'Sparse Optical Flow - Colour Coded ({len(sparse_flow["x"])} points)')
    ax.axis('off')
    fig.tight_layout()

    return fig


def render_sparse_flow_on_frame(
    sparse_flow: dict,
    frame: np.ndarray,
    scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 255),  # Cyan in BGR
    thickness: int = 2,
    tip_length: float = 0.3,
) -> np.ndarray:
    """
    Render sparse flow vectors directly on a frame using OpenCV.

    Args:
        sparse_flow: Dictionary with 'x', 'y', 'u', 'v' arrays
        frame: Background frame (H, W, 3) in RGB or BGR
        scale: Scale factor for arrow lengths
        color: Arrow colour (B, G, R) for BGR or (R, G, B) for RGB
        thickness: Arrow line thickness
        tip_length: Fraction of arrow length for the tip

    Returns:
        Frame with flow vectors drawn (same format as input)
    """
    output = frame.copy()

    # Draw each vector
    for i in range(len(sparse_flow['x'])):
        x, y = int(sparse_flow['x'][i]), int(sparse_flow['y'][i])
        u, v = sparse_flow['u'][i] * scale, sparse_flow['v'][i] * scale

        # Calculate endpoint
        x_end = int(x + u)
        y_end = int(y + v)

        # Draw arrow
        cv2.arrowedLine(
            output,
            (x, y),
            (x_end, y_end),
            color,
            thickness,
            tipLength=tip_length,
        )

    return output


def create_flow_colormap_legend(
    size: Tuple[int, int] = (200, 200),
) -> np.ndarray:
    """
    Create a colour wheel legend for flow visualisation.

    Args:
        size: Size of the legend image (width, height)

    Returns:
        RGB image (H, W, 3) showing colour wheel
    """
    w, h = size
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - 10

    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    dx = x - cx
    dy = y - cy

    # Calculate angle and distance
    angle = np.arctan2(dy, dx)
    dist = np.sqrt(dx**2 + dy**2)

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = np.where(dist <= radius, 255, 0).astype(np.uint8)  # Mask

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add direction labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Right
    cv2.putText(rgb, 'E', (w - 20, cy), font, font_scale, (0, 0, 0), font_thickness)
    # Left
    cv2.putText(rgb, 'W', (5, cy), font, font_scale, (0, 0, 0), font_thickness)
    # Top
    cv2.putText(rgb, 'N', (cx - 10, 15), font, font_scale, (0, 0, 0), font_thickness)
    # Bottom
    cv2.putText(rgb, 'S', (cx - 10, h - 5), font, font_scale, (0, 0, 0), font_thickness)

    return rgb


def save_sparse_flow_visualization(
    sparse_flow: dict,
    output_path: str,
    image_shape: Tuple[int, int],
    background: Optional[np.ndarray] = None,
    plot_type: str = 'quiver',
    **kwargs,
) -> None:
    """
    Save sparse flow visualisation to file.

    Args:
        sparse_flow: Dictionary with 'x', 'y', 'u', 'v' arrays
        output_path: Path to save image
        image_shape: (height, width) of the image
        background: Optional background image
        plot_type: 'quiver' or 'color'
        **kwargs: Additional arguments for plotting functions
    """
    if plot_type == 'quiver':
        fig = plot_sparse_flow_quiver(sparse_flow, image_shape, background, **kwargs)
    elif plot_type == 'color':
        fig = plot_sparse_flow_color(sparse_flow, image_shape, background, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'quiver' or 'color'")

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
