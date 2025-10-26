#!/bin/bash
# Example CLI usage for sparse optical flow processing

# Basic usage - process entire video with default settings (stride=16)
lucidity process video.mp4 \
    --models raft_optical_flow \
    --output ./output/basic

# Process with custom stride for denser sampling
lucidity process video.mp4 \
    --models raft_optical_flow \
    --flow-stride 8 \
    --output ./output/dense

# Process with coarser sampling for faster processing
lucidity process video.mp4 \
    --models raft_optical_flow \
    --flow-stride 32 \
    --output ./output/sparse

# Process specific frame range (useful for testing)
lucidity process video.mp4 \
    --models raft_optical_flow \
    --start-frame 0 \
    --end-frame 100 \
    --output ./output/first_100_frames

# Process with custom mask detection parameters
# (useful if default mask detection fails)
lucidity process video.mp4 \
    --models raft_optical_flow \
    --flow-stride 16 \
    --mask-threshold 40 \
    --mask-method contour \
    --output ./output/custom_mask

# Combine optical flow with depth estimation
lucidity process video.mp4 \
    --models raft_optical_flow,endomust_depth \
    --flow-stride 16 \
    --output ./output/flow_and_depth

# Process time range instead of frame range
lucidity process video.mp4 \
    --models raft_optical_flow \
    --start-time 10.0 \
    --end-time 30.0 \
    --output ./output/time_range
