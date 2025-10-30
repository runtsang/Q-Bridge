"""Classical QCNN hybrid model combining convolution, quanvolution, and fully connected layers.

The model implements a feature extraction pipeline inspired by quantum convolutional neural networks
and includes a quantum-inspired quanvolution filter (implemented as a 2x2 convolution with
random orthogonal weights).  A fully connected layer mimics the behaviour of a parameterised
quantum circuit, enabling easy comparison to the quantum implementation.

The :class:`FastBaseEstimator` provides a lightweight batch evaluator that can optionally add
shot‑like Gaussian noise to the deterministic outputs.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        # Initialise convolution with random orthogonal weights to mimic a quantum kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        with torch.no_grad():
            # Flatten weight to (out_channels, in_channels * kernel_size * kernel_size)
            w = torch.randn(out_channels, in_channels * kernel_size * kernel_size)
            q, _ = torch.linalg.qr(w)
            self.conv.weight.copy_(q.view_as(self.conv.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class FullyConnectedLayer(nn.Module):
    """Simple fully connected layer that emulates a parameterised quantum circuit."""

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.tanh(self.linear(x))


class QCNNHybrid(nn.Module):
    """Hybrid classical QCNN that combines feature map, quanvolution, and fully connected layers."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Feature map: simple linear embedding
        self.feature_map = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU())
        # Quanvolution filter
        self.qfilter = QuanvolutionFilter(in_channels=input_channels, out_channels=4)
        # Fully connected layers
        self.fc1 = FullyConnectedLayer(in_features=4 * 14 * 14, out_features=64)
        self.fc2 = FullyConnectedLayer(in_features=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        features = self.qfilter(x)  # shape: (batch, 4*14*14)
        features = self.fc1(features)
        logits = self.fc2(features)
        return logits.log_softmax(dim=-1)


def QCNN() -> QCNNHybrid:
    """Factory returning the configured :class:`QCNNHybrid`."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "FastEstimator", "FastBaseEstimator", "QCNN"]
