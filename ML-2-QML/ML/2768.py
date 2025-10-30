"""Enhanced hybrid model for binary classification and regression.

This module merges the classical CNN backbone from the original binary
classifier with a quantum‑inspired hybrid head that can operate in
classification or regression mode.  It also ships a lightweight
superposition data generator and a matching PyTorch dataset class
to support regression experiments, thereby unifying the two reference
pairs into a single API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Regression data utilities (from QuantumRegression.py)
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition samples for regression.

    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Input features of shape (samples, num_features).
    y : np.ndarray
        Target values of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple PyTorch dataset for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# Hybrid head (classical approximation of a quantum expectation)
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a sigmoid head."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# ----------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------
class HybridModel(nn.Module):
    """CNN backbone followed by a hybrid head that can perform
    binary classification or regression depending on ``mode``."""
    def __init__(self, mode: str = "classification") -> None:
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)
        if self.mode == "regression":
            self.reg_head = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        if self.mode == "classification":
            probs = self.hybrid(x)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            out = self.hybrid(x)
            return self.reg_head(out).squeeze(-1)


__all__ = [
    "HybridModel",
    "HybridFunction",
    "Hybrid",
    "generate_superposition_data",
    "RegressionDataset",
]
