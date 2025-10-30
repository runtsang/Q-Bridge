"""Classical hybrid binary classifier with a parametric sigmoid head.

This module augments the original seed by adding a dataset generator that
produces superposition‑based features, a binary classification dataset class,
and a unified ``HybridBinaryClassifier`` that can be dropped into any
PyTorch training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data_binary(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate features ``x`` sampled uniformly from [-1, 1] and a binary label
    derived from the sign of a trigonometric function of the sum of the
    features.  This mimics the quantum superposition distribution used in
    the reference seeds while providing a clear binary target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    # Label is 1 if the trigonometric expression is positive, else 0.
    y = (np.sin(angles) + 0.1 * np.cos(2 * angles)) > 0
    return x, y.astype(np.float32)


class BinaryClassificationDataset(Dataset):
    """Dataset that returns a feature vector and a binary label."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data_binary(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Classical hybrid head
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --------------------------------------------------------------------------- #
# Unified classifier
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    CNN followed by a hybrid head.  The head can be switched between a
    classical sigmoid (``use_classical_head=True``) or a quantum expectation
    head (``use_classical_head=False``).  The quantum path must be provided
    by the caller via ``quantum_head``; this keeps the module pure PyTorch
    while still allowing a quantum backend to be plugged in.
    """

    def __init__(self, use_classical_head: bool = True, quantum_head: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if use_classical_head:
            self.hybrid = Hybrid(1, shift=0.0)
        else:
            if quantum_head is None:
                raise ValueError("quantum_head must be provided when using the quantum head.")
            self.hybrid = quantum_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier", "HybridFunction", "Hybrid", "BinaryClassificationDataset"]
