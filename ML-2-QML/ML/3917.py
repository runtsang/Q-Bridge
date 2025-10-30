from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple

class EstimatorQNN__gen229(nn.Module):
    """
    Combined classical estimator that supports both regression and classification.
    Architecture mirrors the original EstimatorQNN feed‑forward regressor and
    the QuantumClassifierModel builder.
    """

    def __init__(self, num_features: int = 2, depth: int = 1, task: str = "regression") -> None:
        """
        Parameters
        ----------
        num_features: int
            Number of input features (default 2).
        depth: int
            Number of hidden layers.
        task: str
            "regression" or "classification". Determines output dimensionality.
        """
        super().__init__()
        self.task = task
        out_dim = 1 if task == "regression" else 2

        layers: list[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # store metadata for compatibility with quantum counterpart
        self.encoding = list(range(num_features))
        self.weight_sizes = [layer.weight.numel() + layer.bias.numel()
                             for layer in self.net.modules()
                             if isinstance(layer, nn.Linear)]
        self.observables = list(range(out_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    def get_encoding(self) -> list[int]:
        return self.encoding

    def get_weight_sizes(self) -> list[int]:
        return self.weight_sizes

    def get_observables(self) -> list[int]:
        return self.observables

__all__ = ["EstimatorQNN__gen229"]
