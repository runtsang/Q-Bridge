"""Hybrid estimator combining classical neural network evaluation and quantum circuit simulation.

This module defines FastHybridEstimator that can evaluate either a pure PyTorch nn.Module
or a torchquantum.QuantumModule. It supports deterministic evaluation, optional Gaussian
shot noise, and integration with a convolutional encoder from Quantum‑NAT.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# Optional torchquantum import
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:  # pragma: no cover
    tq = None
    tqf = None

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QFCModel(nn.Module):
    """Simple CNN followed by a fully connected projection to four features."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


class FastHybridEstimator:
    """Hybrid estimator that can evaluate classical or quantum models.

    If ``model`` is a torch.nn.Module it is executed deterministically.
    If ``model`` is a torchquantum.QuantumModule it is executed on a
    quantum device.  Optional Gaussian shot noise can be added to the
    deterministic outputs to emulate measurement statistics.
    """

    def __init__(self, model: nn.Module) -> None:
        if tq is None and isinstance(model, getattr(tq, "QuantumModule", object)):
            raise ImportError("torchquantum is required for quantum models.")
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables:
            Callables that map model outputs to scalars.
        parameter_sets:
            Sequence of parameter tuples that are fed to the model.
        shots:
            If provided, Gaussian noise with variance 1/shots is added
            to each deterministic output.
        seed:
            Seed for the random number generator.
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
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "QFCModel"]
