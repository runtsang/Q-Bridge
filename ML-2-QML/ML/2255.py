"""Hybrid estimator for classical neural networks with optional noise and QCNN support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a list of floats to a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """
    Evaluate a PyTorch model (or QCNNModel) for batches of inputs and observables.

    Parameters
    ----------
    model : nn.Module, optional
        The model to evaluate.  If ``None`` the estimator will raise an error
        on ``evaluate``.
    noise_shots : int | None, default=None
        If provided, Gaussian noise with variance ``1/shots`` is added to each
        deterministic output to emulate shot noise.
    noise_seed : int | None, default=None
        Seed for the pseudo‑random number generator used for noise.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.model = model
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        if self.model is not None and not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute the expectation value of each observable for every parameter set.

        The method first evaluates the model deterministically, then optionally
        adds Gaussian noise to each result.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        if self.model is None:
            raise ValueError("No model set for evaluation")

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

        if self.noise_shots is not None:
            rng = np.random.default_rng(self.noise_seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / self.noise_shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


# --------------------------------------------------------------------------- #
# QCNN model – classical emulation of the quantum convolutional network
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully connected layers that mirrors the QCNN construction."""

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
    """Factory returning a freshly constructed QCNNModel."""
    return QCNNModel()
