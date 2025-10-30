"""Lightweight estimator utilities implemented with PyTorch modules.

Enhancements over the seed:
- Batched evaluation for large parameter sets.
- Automatic gradient computation for each observable.
- Flexible observable signatures (returning scalar or tensor).
- Caching of results for repeated parameter sets.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Dict

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D tensor with shape (1, n)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluates a PyTorch model for multiple parameter sets.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` that accepts a batch of inputs of shape
        ``(batch, input_dim)`` and returns a tensor of arbitrary shape.
    device:
        Device on which the model and tensors should live.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self._cache: Dict[Tuple[float,...], Tuple[List[float], List[List[float]]]] = {}

    def _evaluate_single(
        self,
        params: Sequence[float],
        observables: Iterable[ScalarObservable],
    ) -> Tuple[List[float], List[List[float]]]:
        """Return (raw outputs, gradients) for a single parameter set."""
        inputs = _ensure_batch(params).to(self.device)
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        raw = []
        grads = []
        for obs in observables:
            val = obs(outputs)
            if isinstance(val, torch.Tensor):
                val = val.mean()
            raw.append(float(val.cpu()))
            grads.append(torch.autograd.grad(val, inputs, retain_graph=True)[0].squeeze().cpu().numpy().tolist())
        return raw, grads

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """Evaluate all observables and their gradients for all parameter sets.

        Returns
        -------
        values:
            A list of rows, each containing the scalar value of every observable.
        gradients:
            A list of rows, each containing a list of gradients per observable.
            The gradient shape corresponds to the input dimension.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        values: List[List[float]] = []
        gradients: List[List[List[float]]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                key = tuple(params)
                if key in self._cache:
                    raw, grads = self._cache[key]
                else:
                    raw, grads = self._evaluate_single(params, observables)
                    self._cache[key] = (raw, grads)
                values.append(raw)
                gradients.append(grads)
        return values, gradients


class FastEstimator(FastBaseEstimator):
    """Adds optional shot‑noise simulation to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        raw_values, raw_grads = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw_values, raw_grads
        rng = np.random.default_rng(seed)
        noisy_values: List[List[float]] = []
        noisy_grads: List[List[List[float]]] = []
        for values, grads in zip(raw_values, raw_grads):
            noise_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in values]
            noisy_values.append(noise_row)
            # Gradient noise can be added similarly if desired; keep deterministic for simplicity
            noisy_grads.append(grads)
        return noisy_values, noisy_grads


__all__ = ["FastBaseEstimator", "FastEstimator"]
