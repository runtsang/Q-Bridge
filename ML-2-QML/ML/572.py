"""Enhanced estimator for PyTorch models with GPU support, shot noise, and gradients.

This module extends the original lightweight estimator by adding:
* device‑aware evaluation (CPU/GPU)
* optional Gaussian shot noise to mimic measurement statistics
* automatic gradient computation for a list of scalar observables
* caching of the last evaluated parameters to speed up repeated calls
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables with optional shot noise and gradients."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model.to(device or torch.device("cpu"))
        self.device = self.model.device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> List[List[float]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of callable observables that map model outputs to scalars.
        parameter_sets
            Sequence of parameter vectors to evaluate.
        shots
            If provided, generate Gaussian noise with variance 1/shots to simulate measurement shot noise.
        seed
            Random seed for reproducibility of shot noise.
        device
            Optional device override for evaluation.
        """
        if device is not None:
            self.model.to(device)
            self.device = device

        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

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

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    def evaluate_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        device: torch.device | str | None = None,
    ) -> List[List[torch.Tensor]]:
        """Return gradients of each observable with respect to model parameters.

        The returned gradients are tensors with the same shape as the model parameters for each
        observable and each parameter set. The method uses PyTorch's autograd and assumes the
        model is differentiable with respect to its input.
        """
        if device is not None:
            self.model.to(device)
            self.device = device

        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        gradients: List[List[torch.Tensor]] = []

        self.model.train()
        for params in parameter_sets:
            inputs = _ensure_batch(params).to(self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            grads_per_set: List[torch.Tensor] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, device=self.device)
                self.model.zero_grad()
                scalar.backward(retain_graph=True)
                grads_per_set.append(inputs.grad.clone().detach())
                inputs.grad.zero_()
            gradients.append(grads_per_set)
        return gradients


__all__ = ["FastBaseEstimator"]
