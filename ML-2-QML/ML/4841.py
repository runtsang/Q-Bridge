"""Hybrid classical estimator with QCNN and sampler support.

The :class:`HybridEstimator` evaluates any PyTorch model over batches of parameters
and can inject Gaussian noise to emulate measurement statistics.  Factory
functions :func:`QCNN` and :func:`SamplerQNN` return ready‑to‑train models
mirroring the quantum counterparts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridEstimator:
    """Evaluate a PyTorch module over batches of parameters.

    Parameters
    ----------
    model : nn.Module
        Any neural network compatible with the FastBaseEstimator API.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return the scalar observation for each parameter set.

        If ``shots`` is provided, the results are perturbed with Gaussian noise
        to emulate measurement statistics.
        """
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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# QCNN inspired classical architecture
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Dense analogue of the quantum convolution stack.

    The architecture mirrors the layers used in the QCNN helper but with
    fully‑connected layers and Tanh activations.  It is designed to be
    interchangeable with the quantum variant for benchmarking.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory that returns a ready‑to‑train :class:`QCNNModel`."""
    return QCNNModel()


# --------------------------------------------------------------------------- #
# Simple softmax sampler network
# --------------------------------------------------------------------------- #
class SamplerModule(nn.Module):
    """Softmax classifier that emulates the QML SamplerQNN."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNN() -> SamplerModule:
    """Factory returning a :class:`SamplerModule`."""
    return SamplerModule()


__all__ = ["HybridEstimator", "QCNNModel", "QCNN", "SamplerModule", "SamplerQNN"]
