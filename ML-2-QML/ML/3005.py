"""Hybrid classical neural network for the Quantum‑NAT family.

The architecture extends the original QFCModel by adding a deeper
convolutional backbone and a flexible embedding layer that can be fed
into a quantum circuit.  A lightweight estimator interface mirrors the
FastBaseEstimator from the QML seed, allowing deterministic and
shot‑noise‑augmented evaluation of any observable on the network
output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    """Broadcast a 1‑D sequence of scalars to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridNATModel(nn.Module):
    """Classical CNN that produces a 4‑dimensional feature vector.

    The network mirrors the original QFCModel but adds an additional
    convolutional block and a linear embedding that matches the 4‑qubit
    quantum encoder in the QML counterpart.  The output is batch‑normed
    to match the quantum module’s post‑measurement scaling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 4‑dimensional embedding for the 4‑qubit encoder
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a 4‑dimensional feature vector."""
        features = self.features(x)
        embedded = self.embed(features)
        return self.norm(embedded)


class FastBaseEstimator:
    """Deterministic estimator for any torch‑nn.Module.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
    ) -> List[List[float]]:
        """Evaluate a list of observables for each parameter set."""
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
    """Estimator that adds Gaussian shot noise to the deterministic output.

    The noise model is a simple normal distribution with variance
    proportional to 1/shots.  This emulates measurement shot noise in
    a quantum device.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Iterable[Iterable[float]],
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


__all__ = ["HybridNATModel", "FastBaseEstimator", "FastEstimator"]
