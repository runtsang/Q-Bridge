"""Unified classical‑quantum hybrid model.

The module implements a feed‑forward network that feeds into a variational quantum circuit
for classification or regression.  The same class structure is reused for a pure
classical baseline, and a quantum‑kernel wrapper is provided for kernel‑based methods.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Classical backbone – dense feed‑forward network
# --------------------------------------------------------------------------- #
class _DenseBackbone(nn.Module):
    """A flexible dense network that can be used as a classical baseline."""
    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None, activation: nn.Module = nn.ReLU()):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation)
            input_dim = h
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# --------------------------------------------------------------------------- #
#  Helper: build classifier circuit (classical analogue)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int, hidden_sizes: list[int] | None = None) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    The return values are a network, an encoding list, a list of weight counts, and
    a list of output observables.
    """
    backbone = _DenseBackbone(num_features, hidden_sizes)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in backbone.parameters()]
    observables = [0]  # single output neuron
    return backbone, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Classical radial basis function kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Regression utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Hybrid head (classical sigmoid with shift)
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
#  Hybrid CNN (classical backbone + hybrid head)
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)

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
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

# --------------------------------------------------------------------------- #
#  Unified model (classical baseline)
# --------------------------------------------------------------------------- #
class UnifiedQuantumHybridModel(nn.Module):
    """
    Unified classical model that can act as a baseline or a hybrid head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_sizes : list[int] | None
        Hidden layer sizes for the dense backbone.
    task : str
        Either 'classification' or'regression'.
    use_cnn : bool
        If True, the model uses the CNN+Hybrid head defined in QCNet.
    """
    def __init__(self, input_dim: int, hidden_sizes: list[int] | None = None,
                 task: str = "classification", use_cnn: bool = False):
        super().__init__()
        self.task = task
        self.use_cnn = use_cnn
        if use_cnn:
            self.base = QCNet()
        else:
            self.backbone = _DenseBackbone(input_dim, hidden_sizes)
            last_dim = self.backbone.layers[-1].out_features
            if task == "classification":
                self.head = Hybrid(last_dim, shift=0.0)
            else:
                self.head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            return self.base(x)
        features = self.backbone(x)
        out = self.head(features)
        if self.task == "classification":
            prob = out
            return torch.cat((prob, 1 - prob), dim=-1)
        return out

__all__ = [
    "build_classifier_circuit",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "generate_superposition_data",
    "RegressionDataset",
    "HybridFunction",
    "Hybrid",
    "QCNet",
    "UnifiedQuantumHybridModel",
]
