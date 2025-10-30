"""Enhanced lightweight estimator utilities built on PyTorch."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D batch tensor from a list of scalar parameters."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """A lightweight, batched PyTorch estimator.

    The class accepts any nn.Module and exposes a ``predict`` method that
    returns a NumPy array of shape (n_samples, n_observables).  The
    inference is performed on the chosen device (CPU or GPU) and the
    outputs are automatically converted to a float‑valued NumPy array.
    """

    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the model on a batch and return the raw output."""
        self.model.eval()
        with torch.no_grad():
            return self.model(batch.to(self.device))

    def predict(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return a matrix of shape (n_samples, n_observables) with
        computed observables.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        # Convert all parameter sets into a single batch tensor
        batch = torch.tensor(parameter_sets, dtype=torch.float32, device=self.device)
        outputs = self._forward(batch)
        results: List[List[float]] = []
        for out in outputs:
            row: List[float] = []
            for obs in observables:
                val = obs(out)
                if isinstance(val, torch.Tensor):
                    scalar = float(val.mean().cpu())
                else:
                    scalar = float(val)
                row.append(scalar)
            results.append(row)
        return np.array(results, dtype=float)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compatibility wrapper that returns a plain Python list."""
        return self.predict(observables, parameter_sets).tolist()


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def predict(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        raw = super().predict(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = rng.normal(loc=raw, scale=1.0 / np.sqrt(shots))
        return noisy

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return self.predict(
            observables,
            parameter_sets,
            shots=shots,
            seed=seed,
        ).tolist()


__all__ = ["FastBaseEstimator", "FastEstimator"]
