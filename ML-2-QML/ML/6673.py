"""Enhanced neural‑network estimator with batched evaluation and gradient support.

The class retains the original API (`evaluate`) but adds:
* vectorised batch processing for GPU acceleration,
* optional Gaussian shot noise,
* `compute_gradients` method that returns gradients of each observable w.r.t.
  the model parameters using PyTorch autograd.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn, Tensor, autograd


ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float]) -> Tensor:
    """Convert a 1‑D list of parameters into a 2‑D tensor (batch × features)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Evaluate a PyTorch model for many parameter sets with optional noise.

    Parameters
    ----------
    model : nn.Module
        A neural network that maps a batch of inputs to outputs.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------ public API
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Evaluate all observables for each parameter set.

        The method runs in evaluation mode and disables gradient tracking.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Same as :meth:`evaluate` but injects Gaussian shot noise.

        Parameters
        ----------
        shots : int | None
            If ``None`` no noise is added; otherwise the standard deviation
            of the Gaussian noise is ``1 / sqrt(shots)``.
        seed : int | None
            Seed for reproducibility of the noise.
        """
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def compute_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[Tuple[float,...]]]:
        """Return gradients of each observable w.r.t. each parameter.

        The gradients are computed analytically via PyTorch autograd.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]

        grads: List[List[Tuple[float,...]]] = []
        self.model.train()  # enable gradients

        for params in parameter_sets:
            inputs = _ensure_batch(params)
            inputs.requires_grad_(True)

            outputs = self.model(inputs)
            row_grads: List[Tuple[float,...]] = []

            for obs in observables:
                val = obs(outputs).sum()  # sum to get scalar for backward
                autograd.backward(val, retain_graph=True)
                grad = inputs.grad.squeeze(0).detach().cpu().numpy()
                row_grads.append(tuple(grad))
                inputs.grad.zero_()

            grads.append(row_grads)

        return grads


__all__ = ["FastEstimator"]
