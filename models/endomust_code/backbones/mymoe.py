"""
Simple MoE (Mixture of Experts) layer implementation for inference.

For inference with pretrained weights, this layer behaves like a normal linear layer.
"""

import torch
import torch.nn as nn


class MoELinear(nn.Module):
    """Mixture of Experts Linear layer for inference."""

    def __init__(self, in_features, out_features, num_experts=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        # For inference, we use a simple linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # For inference, just use the loaded weights
        return self.linear(x)
