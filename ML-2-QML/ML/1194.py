"""
HybridBinaryClassifier – Classical PyTorch model with feature‑map learning and a quantum‑style head.

The model mirrors the structure of the original hybrid network but replaces the quantum
circuit with a differentiable sigmoid head.  A learnable 1×1 convolution acts as a
feature‑map before the dense layers, allowing the network to adapt its representation
to the downstream task.  The final head is a custom autograd function that
applies a sigmoid with an optional shift, mimicking the behaviour of the quantum
expectation value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMap(nn.Module):
    """Learnable 1×1 convolution + ReLU feature map."""
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head with an optional shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Head that forwards activations through the HybridFunction."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN with a feature‑map, dense layers, and a quantum‑style head."""
    def __init__(self):
        super().__init__()
        self.feature_map = FeatureMap(in_channels=3, out_channels=32)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Compute the flattened size after conv layers
        dummy_input = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            x = self.feature_map(dummy_input)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = self.drop1(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = self.drop1(x)
            x = torch.flatten(x, 1)
        self.fc1 = nn.Linear(x.size(1), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FeatureMap", "HybridFunction", "Hybrid", "HybridBinaryClassifier"]
