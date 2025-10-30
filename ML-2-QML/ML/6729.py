"""Hybrid estimator combining classical neural nets and quantum variational circuits."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Callable, Union, Protocol

# ---------------------------------------------------------------------------
# 1.  Protocol for quantum estimators
# ---------------------------------------------------------------------------
class QuantumEstimatorProtocol(Protocol):
    def evaluate(self, observables: Iterable, parameter_sets: List[List[float]]) -> List[List[complex]]:
       ...

# ---------------------------------------------------------------------------
# 2.  Helper to convert 1‑D parameter list to a batched tensor
# ---------------------------------------------------------------------------
def _ensure_batch(values: List[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# ---------------------------------------------------------------------------
# 3.  Classical‑only estimator
# ---------------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a PyTorch model or a quantum estimator for batches of inputs and observables."""
    def __init__(self, model: Union[nn.Module, QuantumEstimatorProtocol]) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor], float] | Callable[[complex], complex]],
        parameter_sets: List[List[float]],
    ) -> List[List[float]]:
        if isinstance(self.model, nn.Module):
            return self._evaluate_torch(observables, parameter_sets)
        else:
            return self._evaluate_quantum(observables, parameter_sets)

    def _evaluate_torch(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor] | Callable[[torch.Tensor], float]],
        parameter_sets: List[List[float]],
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

    def _evaluate_quantum(
        self,
        observables: Iterable,
        parameter_sets: List[List[float]],
    ) -> List[List[float]]:
        # Quantum estimator's evaluate returns complex numbers; cast to float
        raw = self.model.evaluate(observables, parameter_sets)
        return [[float(val.real) for val in row] for row in raw]

# ---------------------------------------------------------------------------
# 4.  Estimator with shot‑noise simulation
# ---------------------------------------------------------------------------
class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: List[List[float]],
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

# ---------------------------------------------------------------------------
# 5.  Hybrid estimator that can combine a PyTorch model with an optional quantum estimator
# ---------------------------------------------------------------------------
class FastHybridEstimator(FastEstimator):
    """Hybrid estimator that can combine a PyTorch model with an optional quantum estimator."""
    def __init__(
        self,
        model: Union[nn.Module, QuantumEstimatorProtocol],
        quantum_estimator: QuantumEstimatorProtocol | None = None,
    ) -> None:
        super().__init__(model)
        self.quantum_estimator = quantum_estimator

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        # Classical part
        classical_results = super().evaluate(observables, parameter_sets, shots=shots, seed=seed)
        if self.quantum_estimator is None:
            return classical_results
        # Quantum part
        quantum_results = self.quantum_estimator.evaluate(observables, parameter_sets)
        # Concatenate results side‑by‑side
        combined: List[List[float]] = []
        for c_row, q_row in zip(classical_results, quantum_results):
            combined.append(c_row + q_row)
        return combined

__all__ = ["FastBaseEstimator", "FastEstimator", "FastHybridEstimator"]
