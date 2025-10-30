"""Enhanced lightweight estimator utilities built on PyTorch.

Provides:
* Batched evaluation on CPU or GPU.
* Customizable observables via callables.
* Optional shot noise emulation.
* Simple interface compatible with the original FastBaseEstimator.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters into a 2‑D torch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for many parameter sets.

    Parameters
    ----------
    model:
        A torch.nn.Module that takes a 2‑D tensor of shape
        (batch_size, num_params) and returns a tensor of shape
        (batch_size, output_dim).
    device:
        Device on which the model is executed. ``'cpu'`` by default.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of observable values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of callables that map the model output to a scalar.
        parameter_sets:
            Sequence of parameter vectors; each vector is applied to the model.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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
    """Adds Gaussian shot‑noise to the deterministic estimator.

    The noise model mimics the standard deviation expected from
    a finite number of shots: ``σ ≈ 1/√shots``.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return noisy estimates of the observables.

        Parameters
        ----------
        shots:
            Number of shots to emulate. If ``None`` the estimator is deterministic.
        seed:
            Random seed for reproducibility.
        """
        deterministic = super().evaluate(observables, parameter_sets)
        if shots is None:
            return deterministic

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in deterministic:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
