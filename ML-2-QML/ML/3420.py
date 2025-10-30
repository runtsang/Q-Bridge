"""Hybrid estimator that supports classical neural networks and quantum‑inspired QFCModel.

The estimator extends the original FastBaseEstimator with optional Gaussian shot noise
and the ability to instantiate the classical QFCModel architecture from Quantum‑NAT.
It remains fully classical, using PyTorch and NumPy, and evaluates arbitrary scalar
observables on the model's output.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate neural networks for batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
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
    """Adds optional Gaussian shot noise to the deterministic estimator."""
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


# Import the classical QFCModel architecture from the Quantum‑NAT example
from.QuantumNAT import QFCModel as ClassicalQFCModel


class FastHybridEstimator(FastEstimator):
    """
    Hybrid estimator that can wrap any nn.Module, including the QFCModel architecture.

    Parameters
    ----------
    model : nn.Module | None
        The neural‑network model. If ``None`` and ``model_type`` is ``'qfc'``,
        a default ClassicalQFCModel is instantiated.
    model_type : str, optional
        ``'classic'`` (default) or ``'qfc'``. Determines whether a default model is
        created when ``model`` is ``None``.
    """
    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_type: str = "classic",
        **model_kwargs,
    ) -> None:
        if model is None:
            if model_type == "qfc":
                model = ClassicalQFCModel()
            else:
                raise ValueError("model must be provided for classic type")
        super().__init__(model)


__all__ = ["FastBaseEstimator", "FastEstimator", "FastHybridEstimator"]
