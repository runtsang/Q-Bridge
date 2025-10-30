"""Enhanced FastBaseEstimator for classical neural network evaluation with caching, GPU support, and optional shot noise."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables.

    Features
    --------
    * GPU/CPU device selection.
    * Result caching to avoid redundant forward passes.
    * Optional Gaussian shot noise to emulate finite‑sample statistics.
    * Gradient evaluation via ``evaluate_with_grad``.
    """

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._cache: dict[tuple[float,...], torch.Tensor] = {}

    def _cached_forward(self, params: Sequence[float]) -> torch.Tensor:
        key = tuple(params)
        if key not in self._cache:
            inputs = _ensure_batch(params, self.device)
            self._cache[key] = self.model(inputs)
        return self._cache[key]

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable accepts the model output tensor and returns a scalar
            (tensor or float).  If empty, the mean of the output is used.
        parameter_sets : sequence of parameter sequences
            Each inner sequence contains the input values for the model.
        shots : int or None, optional
            When provided, Gaussian noise with variance ``1/shots`` is added
            to each observable to mimic finite measurement statistics.
        seed : int or None, optional
            Random seed for reproducibility of shot noise.

        Returns
        -------
        List[List[float]]
            Outer list indexed by parameter set, inner list indexed by observable.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        rng = np.random.default_rng(seed) if shots is not None else None

        for params in parameter_sets:
            outputs = self._cached_forward(params)
            row: List[float] = []
            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            if shots is not None:
                row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            results.append(row)
        return results

    def evaluate_with_grad(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """Evaluate observables and their gradients w.r.t. input parameters.

        Returns
        -------
        Tuple containing:
            - List[List[float]] of observable means.
            - List[List[List[float]]] of gradients:
              outermost list over parameter sets,
              next over observables,
              innermost over input dimensions.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        values: List[List[float]] = []
        grads: List[List[List[float]]] = []

        for params in parameter_sets:
            inputs = _ensure_batch(params, self.device)
            inputs.requires_grad_(True)
            outputs = self.model(inputs)

            row_vals: List[float] = []
            row_grads: List[List[float]] = []

            for obs in observables:
                value = obs(outputs)
                if isinstance(value, torch.Tensor):
                    scalar = value.mean()
                else:
                    scalar = torch.tensor(value, device=self.device, dtype=torch.float32)
                row_vals.append(float(scalar.cpu()))

                grad = torch.autograd.grad(scalar, inputs, retain_graph=True)[0]
                row_grads.append(grad.squeeze(0).detach().cpu().tolist())

            values.append(row_vals)
            grads.append(row_grads)

        return values, grads


__all__ = ["FastBaseEstimator"]
