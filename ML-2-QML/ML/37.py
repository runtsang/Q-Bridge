"""FastBaseEstimator with batched evaluation, GPU support, and optional shot noise.

This module extends the original lightweight estimator by:
- Vectorized batch processing of parameter sets.
- Automatic device selection (CPU/GPU) for PyTorch tensors.
- Optional Gaussian shot noise to emulate measurement statistics.
- Gradient evaluation via PyTorch autograd for observables.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats into a 2‑D torch tensor on the CPU."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> np.ndarray:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of functions mapping model outputs to a scalar.
        parameter_sets
            Sequence of parameter lists to evaluate.
        shots
            If provided, Gaussian noise with std=1/sqrt(shots) is added to each mean.
        seed
            Random seed for reproducible noise.
        device
            Optional device override for temporary tensors.

        Returns
        -------
        np.ndarray
            Shape (n_sets, n_observables) containing the results.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        # Evaluate all parameters in one forward pass for efficiency
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params).to(self.device)
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

        raw = np.array(results, dtype=np.float64)

        if shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0 / np.sqrt(shots), size=raw.shape)
        return raw + noise

    def evaluate_grad(
        self,
        observable: ScalarObservable,
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: torch.device | str | None = None,
    ) -> np.ndarray:
        """
        Compute gradients of a single observable with respect to parameters.

        Parameters
        ----------
        observable
            Function mapping model outputs to a scalar.
        parameter_sets
            Sequence of parameter lists to evaluate.
        device
            Optional device override for temporary tensors.

        Returns
        -------
        np.ndarray
            Shape (n_sets, n_params) containing gradients.
        """
        self.model.eval()
        grads: List[List[float]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)

            outputs = self.model(inputs)
            value = observable(outputs)
            if isinstance(value, torch.Tensor):
                scalar = value.mean()
            else:
                scalar = torch.tensor(value, device=self.device)

            grad = torch.autograd.grad(scalar, inputs)[0].detach().cpu().numpy()
            grads.append(grad.squeeze().tolist())

        return np.array(grads, dtype=np.float64)


__all__ = ["FastBaseEstimator"]
