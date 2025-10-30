"""Hybrid classical estimator for PyTorch models with optional shot noise.

This module extends FastBaseEstimator to support:
- Efficient batch evaluation of any nn.Module.
- Optional Gaussian shot noise to simulate measurement uncertainty.
- Compatibility with the QFCModel architecture from QuantumNAT.
"""

import torch
from torch import nn
import numpy as np
from typing import Iterable, Sequence, List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of input parameters.

    Parameters
    ----------
    model : nn.Module
        Any deterministic PyTorch network.  The model must accept a tensor
        of shape ``(batch,...)`` and return a tensor of shape ``(batch,...)``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Compute scalar observables for each parameter set.

        Parameters
        ----------
        observables : Iterable[Callable]
            Callables that map the model output to a scalar or a tensor.
            If empty, the mean of the outputs is returned.
        parameter_sets : Sequence[Sequence[float]]
            Sequence of parameter vectors to evaluate.

        Returns
        -------
        List[List[float]]
            Nested list with shape ``(len(parameter_sets), len(observables))``.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
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

        return results


class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""

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
