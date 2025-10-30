"""Enhanced fast estimator using PyTorch with gradient support and noise injection.

This module defines FastBaseEstimator and FastEstimator with additional
capabilities:
* batched evaluation with arbitrary device placement.
* vectorized observable computation via callable chains.
* automatic gradient computation via torch.autograd.
* configurable Gaussian shot noise with per‑observable variance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float], device: torch.device | str | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model that maps a 1‑D parameter vector to an output tensor.
    device: str | torch.device | None, optional
        Device on which to perform computations. Defaults to ``torch.device('cpu')``.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        The method accepts a sequence of callable observables; each receives
        the model output and returns a scalar tensor or Python float.  If no
        observables are supplied a default mean over the last dimension is used.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        parameter_sets = list(parameter_sets) or []

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params, device=self.device)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        """Return gradients of each observable with respect to the input parameters.

        Each returned tensor has the same shape as the parameter vector and
        dtype ``torch.float32``.  This method is useful for gradient‑based
        optimization or sensitivity analysis.
        """
        observables = list(observables)
        grads: List[List[torch.Tensor]] = []

        self.model.train()  # enable gradients
        for params in parameter_sets:
            inputs = _ensure_batch(params, device=self.device).requires_grad_(True)
            outputs = self.model(inputs)
            row_grads: List[torch.Tensor] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    grad = torch.autograd.grad(value.sum(), inputs, retain_graph=True)[0]
                else:
                    # scalar float: treat as constant
                    grad = torch.zeros_like(inputs)
                row_grads.append(grad.squeeze(0).detach().cpu())
            grads.append(row_grads)
        return grads

class FastEstimator(FastBaseEstimator):
    """Adds configurable Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        noise_std: float | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            std = noise_std if noise_std is not None else max(1e-6, 1 / shots)
            noisy_row = [float(rng.normal(mean, std)) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastBaseEstimator", "FastEstimator"]
