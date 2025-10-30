"""Hybrid classical QCNN combining convolutional feature extraction and quantum‑inspired layers.

The model consists of:
- A classical 2×2 convolutional filter (QuanvolutionFilter) that extracts local features.
- A stack of fully‑connected layers that emulate quantum convolution and pooling (QCNNHead).
- An optional FastEstimator wrapper that adds Gaussian shot noise to predictions.

This design allows side‑by‑side comparison with the quantum implementation while keeping the
interface identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Sequence, Callable

# Simple 2×2 convolutional filter (quantum‑inspired)
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# Classical QCNN head that mimics the quantum convolution steps
class QCNNHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Hybrid model that stitches the filter and head together
class HybridQCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.head = QCNNHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        return self.head(features)

# Lightweight estimator utilities (FastBaseEstimator + FastEstimator)
class FastBaseEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        import numpy as np
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

def QCNN() -> HybridQCNN:
    """Factory returning the configured HybridQCNN model."""
    return HybridQCNN()

__all__ = ["HybridQCNN", "QCNN", "FastBaseEstimator", "FastEstimator"]
