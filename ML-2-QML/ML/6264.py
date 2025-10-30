"""Enhanced neural‑network estimator for batched inputs and gradient queries.

The original FastBaseEstimator was a minimal, deterministic evaluator.
This extension introduces:
* device‑aware construction (CPU/GPU)
* batched evaluation of multiple observables
* optional gradient computation via PyTorch autograd
* caching of parameter tensors for repeated calls
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of parameters into a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model on batches of parameters.

    Parameters
    ----------
    model : nn.Module
        The neural network to evaluate.
    device : str | torch.device, optional
        Device to run the model on; defaults to CPU.
    """

    def __init__(self, model: nn.Module, device: str | torch.device | None = None) -> None:
        self.model = model
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self._cached_params: torch.Tensor | None = None

    def _prepare_inputs(self, parameter_sets: Sequence[Sequence[float]]) -> torch.Tensor:
        """Cache and return a batch tensor on the target device."""
        if self._cached_params is None or self._cached_params.shape[0]!= len(parameter_sets):
            batch = _ensure_batch(parameter_sets)
            self._cached_params = batch.to(self.device)
        return self._cached_params

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Deterministic evaluation of all observables for every parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(parameter_sets)
            outputs = self.model(inputs)

            for params, out in zip(parameter_sets, outputs):
                row: List[float] = []
                for observable in observables:
                    val = observable(out)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_gradients(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> Tuple[List[List[float]], List[List[torch.Tensor]]]:
        """Return observables and their gradients w.r.t. the input parameters."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        grads: List[List[torch.Tensor]] = []

        self.model.train()  # enable gradients
        inputs = self._prepare_inputs(parameter_sets)
        inputs.requires_grad_(True)

        outputs = self.model(inputs)

        for idx, params in enumerate(parameter_sets):
            row_vals: List[float] = []
            row_grads: List[torch.Tensor] = []

            for observable in observables:
                val = observable(outputs[idx])
                if isinstance(val, torch.Tensor):
                    scalar = val.mean()
                else:
                    scalar = torch.tensor(val, dtype=outputs.dtype, device=self.device)
                row_vals.append(float(scalar.cpu()))
                scalar.backward(retain_graph=True)
                row_grads.append(inputs.grad.clone())
                inputs.grad.zero_()

            results.append(row_vals)
            grads.append(row_grads)

        return results, grads


class FastEstimator(FastBaseEstimator):
    """Deterministic estimator with optional shot‑based Gaussian noise."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimator", "FastEstimator"]
