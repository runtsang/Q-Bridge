"""Hybrid fast estimator combining classical PyTorch models and quantum-inspired noise handling.

The class ``HybridFastEstimator`` extends the original FastBaseEstimator by adding
shot‑noise simulation and a convenient API for evaluating multiple observables
on a batch of parameter sets.  It is designed to be drop‑in compatible with
the legacy FastBaseEstimator while providing a unified interface for both
classical and quantum models.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class HybridFastEstimator:
    """Evaluate a PyTorch model for a list of input parameter sets.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module that accepts a batch of inputs and returns a tensor.
    noise_std : float | None, optional
        Standard deviation of Gaussian shot noise added to each output.
        If ``None`` (default), no noise is added.
    """

    def __init__(self, model: nn.Module, noise_std: Optional[float] = None) -> None:
        self.model = model
        self.noise_std = noise_std

    def _evaluate_det(
        self,
        observables: Iterable[ScalarObservable],
        params: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic forward pass."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for p in params:
                inputs = _ensure_batch(p)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate and optionally add shot noise.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is fed to the model as a batch.
        shots : int | None, optional
            If provided, Gaussian noise with variance ``1/shots`` is added.
        seed : int | None, optional
            Random seed for reproducibility.
        """
        raw = self._evaluate_det(observables, parameter_sets)
        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    @classmethod
    def from_model(cls, model: nn.Module, noise_std: Optional[float] = None) -> "HybridFastEstimator":
        """Convenience constructor."""
        return cls(model, noise_std=noise_std)


__all__ = ["HybridFastEstimator"]
