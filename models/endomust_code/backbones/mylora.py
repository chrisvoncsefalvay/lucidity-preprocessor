"""
Simple LoRA layer implementations for inference.

For inference with pretrained weights, these layers behave like normal linear layers
since the LoRA adaptations are already baked into the loaded weights.
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Standard LoRA Linear layer for inference."""

    def __init__(self, in_features, out_features, r=4, lora_alpha=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha

        # Create the base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices (for loading pretrained weights)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

    def forward(self, x):
        # For inference, just use the loaded weights
        result = self.linear(x)

        # Add LoRA contribution if present
        if self.lora_A is not None and self.lora_B is not None:
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling

        return result


class DVLinear(nn.Module):
    """DV-LoRA (Dynamic Vector LoRA) layer for inference."""

    def __init__(self, in_features, out_features, r=4, lora_alpha=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha

        # Create the base linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices (for loading pretrained weights)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.scaling = lora_alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

    def forward(self, x):
        # For inference, just use the loaded weights
        result = self.linear(x)

        # Add LoRA contribution if present
        if self.lora_A is not None and self.lora_B is not None:
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling

        return result
