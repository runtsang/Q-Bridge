"""Unified classical model combining fully‑connected, CNN, and regression heads with FastEstimator support."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence, Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedFCL(nn.Module):
    """
    Classical hybrid architecture that merges:
    * A fully‑connected layer (as in FCL.py) for scalar inputs.
    * A shallow CNN with global pooling (from QuantumNAT.py) for image‑like data.
    * A regression head (from QuantumRegression.py) mapping to a scalar target.
    The module can be wrapped by FastBaseEstimator for batched evaluation.
    """

    def __init__(self, input_dim: int = 1, img_channels: int = 1, n_features: int = 4) -> None:
        super().__init__()
        # linear branch
        self.linear = nn.Linear(input_dim, 1)

        # convolutional branch
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_cnn = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )
        self.norm = nn.BatchNorm1d(n_features)

        # regression head
        self.head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x could be (batch, features) or (batch, channels, H, W)
        if x.dim() == 2:  # fully connected branch
            lin_out = F.tanh(self.linear(x))
            return lin_out.mean(dim=1)
        elif x.dim() == 4:  # image branch
            features = self.cnn(x)
            flat = features.view(features.size(0), -1)
            fc_out = self.fc_cnn(flat)
            norm_out = self.norm(fc_out)
            return self.head(norm_out).squeeze(-1)
        else:
            raise ValueError("Unsupported input dimensionality")

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate the model on a batch of parameter sets using the FastBaseEstimator pattern."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                out = self(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


__all__ = ["UnifiedFCL"]
