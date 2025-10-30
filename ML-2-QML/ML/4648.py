"""Hybrid classical classifier inspired by quantum interfaces.

The module defines a fully‑connected neural network that mimics the
interface of the original QuantumClassifierModel while embedding a
quantum‑style fully connected layer implemented as a lightweight
parameterized function.  The class is fully compatible with PyTorch
training pipelines and exposes the same metadata (encoding, weight
sizes, observables) that the quantum implementation expects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Utility – synthetic data generation
# --------------------------------------------------------------------------- #

def generate_classification_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a toy classification dataset.

    Samples are drawn from a uniform distribution.  The label is 1 if
    sin(sum(x)) > 0 else 0.  This mirrors the superposition data used
    in the regression example but with a binary target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    y = (np.sin(np.sum(x, axis=1)) > 0).astype(np.int64)
    return x, y

class ClassificationDataset(Dataset):
    """Torch Dataset wrapping the synthetic classification data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classification_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.long),
        }

# --------------------------------------------------------------------------- #
# Classical “quantum” layer – inspired by the FCL example
# --------------------------------------------------------------------------- #

def FCL(num_features: int = 1) -> nn.Module:
    """
    A lightweight stand‑in for a quantum fully‑connected layer.
    It accepts a vector of parameters and returns the mean of a
    non‑linear transformation, mimicking an expectation value.
    """
    class QuantumLikeLayer(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.linear = nn.Linear(n_features, 1, bias=False)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            # Use tanh to emulate an expectation of a bounded observable
            expectation = torch.tanh(self.linear(theta_tensor)).mean(dim=0)
            return expectation.detach().cpu().numpy()

    return QuantumLikeLayer(num_features)

# --------------------------------------------------------------------------- #
# Hybrid classifier – classical core + optional quantum layer
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Build a feed‑forward network that mirrors the structure of the
    quantum circuit: a stack of linear layers with ReLU activations,
    followed by a head that outputs two logits.
    Returns the network together with metadata expected by the quantum
    side (encoding indices, weight sizes, observables).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=True)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))  # placeholder for two class logits
    return network, encoding, weight_sizes, observables

class HybridQuantumClassifier(nn.Module):
    """
    Classical neural network that mimics a quantum classifier interface.
    It optionally attaches a lightweight quantum‑style fully‑connected
    layer (FCL) after the classical backbone.
    """
    def __init__(self, num_features: int, depth: int = 2, use_fcl: bool = True):
        super().__init__()
        self.backbone, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.use_fcl = use_fcl
        self.fcl = FCL(num_features) if use_fcl else None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical backbone and, when enabled,
        the quantum‑style layer.  The final logits are the sum of both
        contributions, emulating a hybrid quantum‑classical model.
        """
        logits = self.backbone(states)
        if self.use_fcl and self.fcl is not None:
            # Treat each sample as a set of parameters for the FCL
            fcl_out = torch.from_numpy(
                np.vstack([self.fcl.run(state.tolist()) for state in states])
            ).squeeze(-1)
            logits = logits + fcl_out.unsqueeze(-1)
        return logits

__all__ = ["HybridQuantumClassifier", "ClassificationDataset", "generate_classification_data"]
