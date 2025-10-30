"""Enhanced estimator utilities built on PyTorch.

Provides a batch‑aware base estimator and a noisy variant that can emulate
shot‑limited measurements.  The API is intentionally compatible with the
original FastBaseEstimator while adding device support, vectorised
evaluation, and a convenient dictionary‑style observable interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets.

    Parameters
    ----------
    model
        A ``torch.nn.Module`` that maps a 1‑D tensor of parameters to an
        output tensor.  The module is moved to ``device`` on construction.
    device
        Target device (CPU or CUDA).  If ``None`` the model's current device
        is used.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> List[List[float]]:
        """Return scalar results for each observable and parameter set.

        The computation is performed in batches to keep memory usage bounded.
        """
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            # Convert to tensor once to avoid repeated conversions
            param_tensor = torch.as_tensor(
                parameter_sets, dtype=torch.float32, device=self.device
            )

            for start in range(0, param_tensor.shape[0], batch_size):
                batch = param_tensor[start : start + batch_size]
                outputs = self.model(batch)

                for row in range(batch.shape[0]):
                    row_values: List[float] = []
                    for observable in observables:
                        value = observable(outputs[row])
                        if isinstance(value, torch.Tensor):
                            scalar = float(value.mean().cpu())
                        else:
                            scalar = float(value)
                        row_values.append(scalar)
                    results.append(row_values)

        return results

    def evaluate_named(
        self,
        observables: Dict[str, ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> List[Dict[str, float]]:
        """Convenience wrapper that returns a list of dicts keyed by observable name."""
        names = list(observables.keys())
        obs_list = list(observables.values())
        raw = self.evaluate(obs_list, parameter_sets, batch_size)
        return [{name: val for name, val in zip(names, row)} for row in raw]


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator.

    The noise is added after the deterministic evaluation, mimicking
    measurement statistics for a finite number of shots.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        batch_size: int = 64,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, batch_size)
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
