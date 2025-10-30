"""Enhanced PyTorch estimator with device abstraction, batched evaluation, and optional gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn, Tensor

# type alias for a callable that turns model outputs into a scalar
ScalarObservable = Callable[[Tensor], Tensor | float]


def _ensure_batch(values: Sequence[float], device: torch.device) -> Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor on the requested device."""
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen:
    """Base estimator that evaluates a PyTorch model for many parameter sets.

    Parameters
    ----------
    model : nn.Module
        A callable neural network that maps a batch of parameters to output vectors.
    device : torch.device, optional
        Target device for tensors. Defaults to ``'cpu'``.
    """

    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        return_gradients: bool = False,
    ) -> List[List[float]]:
        """Return a matrix of observable values for each parameter set.

        If ``return_gradients`` is True, the method also computes the gradient of each
        observable with respect to the input parameters and returns it as a flattened
        list appended to each row.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        for params in parameter_sets:
            inputs = _ensure_batch(params, self.device)
            if return_gradients:
                inputs.requires_grad_(True)
            outputs = self.model(inputs)
            row: List[float] = []
            for observable in observables:
                value = observable(outputs)
                if isinstance(value, Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
                if return_gradients:
                    # compute gradient of the scalar w.r.t. input parameters
                    scalar.backward(retain_graph=True)
                    grad = inputs.grad.squeeze(0).detach().cpu().numpy().tolist()
                    row.extend(grad)
            if return_gradients:
                # reset gradients for the next iteration
                inputs.grad.zero_()
            results.append(row)
        return results


class FastEstimatorGen(FastBaseEstimatorGen):
    """Extends the base estimator with optional shot‑noise emulation."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
        return_gradients: bool = False,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets, return_gradients=return_gradients)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            # the deterministic part occupies the first len(observables) entries
            means = row[: len(observables)]
            stds = [max(1e-6, 1 / shots) for _ in means]
            noisy_row = [
                float(rng.normal(m, s)) for m, s in zip(means, stds)
            ]
            # keep gradient entries unchanged
            noisy_row.extend(row[len(observables) :])
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastBaseEstimatorGen", "FastEstimatorGen"]
