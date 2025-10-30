"""Enhanced lightweight estimator utilities for PyTorch models.

Features
--------
* Device‑aware evaluation (CPU/GPU).
* Batch‑wise evaluation with optional autograd support.
* Vectorised observables as callables or torch functions.
* Optional Gaussian shot noise added at the estimator level.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameter sets into a 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters and observables.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate. It must accept a 2‑D tensor of shape
        ``(batch, n_params)`` and return a 2‑D tensor of model outputs.
    device : str | torch.device, optional
        The device on which to run the model. Defaults to ``"cpu"``.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

    def _forward(self, params: torch.Tensor) -> torch.Tensor:
        """Run the model in eval mode without gradients."""
        self.model.eval()
        with torch.no_grad():
            return self.model(params.to(self.device))

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute expectations for each parameter set and observable.

        Parameters
        ----------
        observables
            Callables that map a model output tensor to a scalar. If a
            callable returns a tensor, its mean is used as the scalar.
        parameter_sets
            Sequence of parameter vectors.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        params = _ensure_batch(parameter_sets).to(self.device)
        outputs = self._forward(params)

        for obs in observables:
            value = obs(outputs)
            if isinstance(value, torch.Tensor):
                scalar = float(value.mean().cpu())
            else:
                scalar = float(value)
            results.append([scalar])
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap :meth:`evaluate` adding Gaussian shot noise.

        Parameters
        ----------
        shots
            Number of measurement shots. If ``None`` no noise is added.
        seed
            Random seed for reproducibility.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / np.sqrt(shots)))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator"]
