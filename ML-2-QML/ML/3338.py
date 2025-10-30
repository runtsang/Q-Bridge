"""Hybrid classical estimator mirroring the quantum ansatz structure.

The model supports regression or binary classification by selecting
the output head and activation function.  Its internal layers match
the parameter layout of the quantum circuit: an encoding layer,
followed by `depth` blocks of linear + non‑linearity, and a task‑
specific output layer.  The class exposes weight sizes and observable
information to facilitate joint quantum‑classical experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, List

class HybridEstimatorQNN(nn.Module):
    def __init__(self, num_features: int = 2, depth: int = 1, task: str = "regression") -> None:
        super().__init__()
        self.task = task
        self.num_features = num_features
        self.depth = depth

        # Encoding layer (identity) to match quantum input_params
        self.encoding = nn.Identity()

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            act = nn.Tanh() if task == "regression" else nn.ReLU()
            layers.extend([lin, act])
            in_dim = num_features

        out_dim = 1 if task == "regression" else 2
        head = nn.Linear(in_dim, out_dim)
        layers.append(head)

        self.net = nn.Sequential(*layers)

        # Record weight sizes to align with quantum weight vector
        self.weight_sizes: List[int] = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                self.weight_sizes.append(module.weight.numel() + module.bias.numel())

        # Observable mapping
        self.observables = ["Y"] if task == "regression" else ["Z"] * out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoding(x)
        return self.net(x)

    def get_encoding_params(self) -> Iterable[str]:
        """Return names of parameters that correspond to the classical encoding."""
        return []

    def get_weight_params(self) -> Iterable[str]:
        """Return names of parameters that correspond to the classical weights."""
        return [f"module.{i}.weight" for i in range(len(self.weight_sizes))]

def EstimatorQNN(num_features: int = 2, depth: int = 1, task: str = "regression") -> HybridEstimatorQNN:
    """Return a hybrid estimator instance with default configuration."""
    return HybridEstimatorQNN(num_features, depth, task)

__all__ = ["EstimatorQNN", "HybridEstimatorQNN"]
