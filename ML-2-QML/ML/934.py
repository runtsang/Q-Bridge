"""Enhanced classical‑only classifier that mirrors the hybrid architecture but adds a residual skip‑connection and a multi‑head dense block."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualHybridFunction(torch.autograd.Function):
    """Custom autograd function that forwards through a dense head and returns a sigmoid
    activation while preserving gradient propagation to the original input."""
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, shift: float) -> torch.Tensor:
        logits = input_tensor.sum(dim=1, keepdim=True)
        output = torch.sigmoid(logits + shift)
        ctx.save_for_backward(input_tensor)
        ctx.shift = shift
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input_tensor, = ctx.saved_tensors
        shift = ctx.shift
        logits = input_tensor.sum(dim=1, keepdim=True)
        s = torch.sigmoid(logits + shift)
        grad_input = grad_output * s * (1 - s)
        grad_input = grad_input.expand_as(input_tensor)
        return grad_input, None

class QuantumHybridClassifier(nn.Module):
    """Classical residual network with a hybrid quantum expectation head."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.skip = nn.Linear(84, 1)
        self.shift = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x_fc2 = F.relu(self.fc2(x))
        skip_out = self.skip(x_fc2)
        x_fc3 = self.fc3(x_fc2)
        x = x_fc3 + skip_out
        probs = ResidualHybridFunction.apply(x, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ResidualHybridFunction", "QuantumHybridClassifier"]
