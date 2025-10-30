"""Hybrid classical regression model incorporating convolutional filtering and quantum-inspired layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Import the classical Conv filter
from.Conv import Conv as ConvFilter


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from a superposition‑like function, optionally reshaped for Conv filtering."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that optionally applies a classical Conv filter to each sample."""

    def __init__(self, samples: int, num_features: int, use_conv: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = ConvFilter(kernel_size=2, threshold=0.0)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat = self.features[idx]
        if self.use_conv:
            # Reshape to a square if possible for the Conv filter
            size = int(np.sqrt(len(feat)))
            if size * size == len(feat):
                feat = feat.reshape(size, size)
                conv_out = self.conv.run(feat)
                feat = np.array([conv_out], dtype=np.float32)
        return {
            "states": torch.tensor(feat, dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head mimicking a quantum expectation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        outputs, = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class QuantumRegressionModel(nn.Module):
    """Classical pre‑processing followed by a hybrid quantum‑style head."""

    def __init__(self, num_features: int, num_qubits: int = 4):
        super().__init__()
        # Classical dense pre‑processing
        self.pre = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_qubits),
            nn.ReLU(),
        )
        # Quantum‑style expectation head
        self.shift = 0.0

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        z = self.pre(states)
        # Use the differentiable hybrid function as the quantum expectation head
        return HybridFunction.apply(z, self.shift)


__all__ = ["RegressionDataset", "QuantumRegressionModel", "generate_superposition_data"]
