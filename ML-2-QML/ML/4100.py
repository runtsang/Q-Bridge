"""Classical hybrid CNN with a quantum‑inspired head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class FCLayer(nn.Module):
    """Classical surrogate for a quantum fully‑connected layer."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # emulate quantum expectation: tanh + mean
        return torch.tanh(self.linear(x)).mean(dim=-1, keepdim=True)

class Hybrid(nn.Module):
    """Hybrid head that blends a classical linear layer with a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.fc = FCLayer(in_features)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return HybridFunction.apply(logits, self.shift)

class HybridQuantumCNN(nn.Module):
    """CNN followed by a hybrid head that emulates the quantum expectation."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=shift)

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
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def evaluate_with_noise(self, inputs: torch.Tensor, shots: int, seed: int | None = None) -> torch.Tensor:
        """Return noisy predictions using Gaussian shot‑noise."""
        with torch.no_grad():
            probs = self.forward(inputs)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1 / np.sqrt(shots), size=probs.shape)
            return probs + noise

__all__ = ["HybridFunction", "FCLayer", "Hybrid", "HybridQuantumCNN"]
