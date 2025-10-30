"""HybridEstimator: classical estimator with optional quantum transformer.

The module defines a hybrid estimator that extends the classical
FastBaseEstimator to optionally use a quantum transformer for
evaluation.  The implementation keeps the original FastBaseEstimator
API but adds a ``quantum`` flag that triggers a quantum‑based
forward pass if the model contains a quantum submodule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Any

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor for a batch of parameter vectors."""
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


class HybridEstimator(FastBaseEstimator):
    """Hybrid estimator that can optionally use a quantum transformer.

    Parameters
    ----------
    model : nn.Module
        A PyTorch model that may contain a quantum transformer submodule.
    quantum : bool, optional
        If True, the estimator will attempt to use the quantum transformer
        for evaluation.  If the model does not contain a quantum module,
        a ValueError is raised.
    """

    def __init__(self, model: nn.Module, *, quantum: bool = False) -> None:
        super().__init__(model)
        self.quantum = quantum

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Run the model in either classical or quantum mode."""
        if self.quantum:
            return self._evaluate_quantum(observables, parameter_sets, shots=shots, seed=seed)
        return super().evaluate(observables, parameter_sets)

    def _evaluate_quantum(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Internal helper that evaluates using a quantum transformer."""
        # Search for a submodule that represents a quantum transformer
        quantum_module = None
        for name, module in self.model.named_children():
            if name == "quantum_transformer" or name.endswith("Quantum"):
                quantum_module = module
                break
        if quantum_module is None:
            raise ValueError("No quantum transformer found in the model.")
        quantum_module.eval()
        inputs = _ensure_batch(parameter_sets)
        with torch.no_grad():
            quantum_outputs = quantum_module(inputs)
        # Evaluate observables on the quantum outputs
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        for out in quantum_outputs:
            row: List[float] = []
            for observable in observables:
                value = observable(out)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)
        return results
