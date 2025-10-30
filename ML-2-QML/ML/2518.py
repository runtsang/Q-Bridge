from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List

import qiskit
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class FastHybridEstimator:
    """Hybrid estimator that can evaluate either a classical PyTorch model or a Qiskit circuit."""
    def __init__(self,
                 model: nn.Module | None = None,
                 circuit: 'qiskit.circuit.QuantumCircuit' | None = None) -> None:
        if model is None and circuit is None:
            raise ValueError("Either a PyTorch model or a Qiskit circuit must be supplied.")
        self.model = model
        self.circuit = circuit

    def _evaluate_classical(self,
                            observables: Iterable[ScalarObservable],
                            parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
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

    def _evaluate_quantum(self,
                          observables: Iterable[BaseOperator],
                          parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        from qiskit.quantum_info import Statevector
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind_circuit(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind_circuit(self, parameter_values: Sequence[float]):
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable,
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float | complex]]:
        if self.model is not None:
            raw = self._evaluate_classical(observables, parameter_sets)
            if shots is None:
                return raw
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in raw:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy
        else:
            raw = self._evaluate_quantum(observables, parameter_sets)
            return raw

__all__ = ["FastHybridEstimator", "HybridFunction", "Hybrid"]
