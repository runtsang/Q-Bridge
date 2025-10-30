"""Hybrid estimator combining classical neural networks with optional quantum layers.

The class accepts a PyTorch nn.Module or a list of modules and evaluates
observables for a sequence of parameter sets.  Gaussian shot noise can be
added to mimic quantum measurement statistics.  Helper factories provide
common hybrid architectures such as a quanvolution filter followed by a
linear head, a classical convolution filter, and a fully‑connected quantum
layer implemented with a simple Qiskit circuit.

The implementation is intentionally lightweight and relies only on
NumPy, PyTorch and standard library modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch model for batches of inputs and observables.

    Parameters
    ----------
    model
        A single ``nn.Module`` or a list of modules that will be
        applied sequentially.  Each module must accept a 2‑D tensor of shape
        ``(batch, features)`` and return a tensor of the same shape.
    noise
        If ``True`` Gaussian shot noise is added to the deterministic
        predictions.  The ``shots`` argument controls the variance.
    shots
        Number of shots used to generate the noise.  ``None`` disables noise.
    seed
        Random seed for reproducibility of the noise.
    """

    def __init__(
        self,
        model: Union[nn.Module, Sequence[nn.Module]],
        *,
        noise: bool = False,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(model, nn.Module):
            self.model = model
        else:
            self.model = nn.Sequential(*model)
        self.noise = noise
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a matrix of observable values.

        Each row corresponds to a parameter set and each column to an
        observable.  If ``self.noise`` is ``True`` Gaussian noise is added
        according to ``self.shots``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)

                if self.noise and self.shots:
                    row = [
                        float(self.rng.normal(mean, max(1e-6, 1 / self.shots)))
                        for mean in row
                    ]
                results.append(row)

        return results


# --------------------------------------------------------------------------- #
# Helper architectures
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """2‑D convolution that emulates a simple quantum kernel."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid network: quanvolution filter → linear head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        # 4 * 14 * 14 = 784 features for 28×28 MNIST images
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return torch.nn.functional.log_softmax(logits, dim=-1)


class ConvFilter(nn.Module):
    """Drop‑in classical convolution filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])


class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected layer that mimics a quantum fully‑connected layer."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        expectation = torch.tanh(self.linear(thetas)).mean(dim=0)
        return expectation


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #

def create_quanvolution_classifier() -> nn.Module:
    """Return a ready‑to‑use quanvolution classifier."""
    return QuanvolutionClassifier()


def create_conv_filter() -> nn.Module:
    """Return a classical convolution filter."""
    return ConvFilter()


def create_fcl() -> nn.Module:
    """Return a simple fully‑connected layer."""
    return FullyConnectedLayer()


__all__ = [
    "HybridEstimator",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "ConvFilter",
    "FullyConnectedLayer",
    "create_quanvolution_classifier",
    "create_conv_filter",
    "create_fcl",
]
