"""Classical QCNN model with integrated fast batch evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List, Sequence

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class QCNNGen187(nn.Module):
    """Enhanced convolution‑inspired neural network.

    The architecture mirrors the quantum convolution steps but introduces
    ReLU, batch‑norm and dropout to improve generalisation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.BatchNorm1d(16)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.1)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8), nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4), nn.ReLU()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Batch‑wise evaluation of the network with optional observables."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                x = _ensure_batch(params)
                out = self(x)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results

def create_qcnn_gen187() -> QCNNGen187:
    """Factory returning an instantiated QCNNGen187."""
    return QCNNGen187()

__all__ = ["QCNNGen187", "create_qcnn_gen187"]
