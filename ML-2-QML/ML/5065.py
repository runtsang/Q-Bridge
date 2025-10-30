from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union, Any

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _apply_shot_noise(values: List[float], shots: int, rng: np.random.Generator) -> List[float]:
    """Add Gaussian shot noise consistent with a finite number of shots."""
    std = max(1e-6, 1 / np.sqrt(shots))
    return [float(rng.normal(v, std)) for v in values]


class HybridFastEstimator:
    """
    Lightweight estimator that can evaluate either a classical PyTorch network
    or a parametrised Qiskit circuit (or any callable that returns one).

    Parameters
    ----------
    model
        * ``torch.nn.Module`` – a classical network
        * ``qiskit.circuit.QuantumCircuit`` – a quantum circuit
        * ``Callable`` – any function that given a parameter list returns
          a circuit (or a hybrid module)
    shots
        Optional number of shots to emulate measurement noise. When ``None``
        the estimator is deterministic.
    seed
        Random seed for the shot‑noise generator.
    """
    def __init__(self,
                 model: Union[nn.Module, QuantumCircuit, Callable[..., Any]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None):
        self.model = model
        self.shots = shots
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self._detect_type()

    def _detect_type(self):
        if isinstance(self.model, nn.Module):
            self._type = "torch"
        elif isinstance(self.model, QuantumCircuit):
            self._type = "qiskit"
        elif callable(self.model):
            self._type = "callable"
        else:
            raise TypeError(f"Unsupported model type: {type(self.model)}")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound."""
        if not isinstance(self.model, QuantumCircuit):
            raise RuntimeError("Attempted to bind parameters to a non‑circuit model.")
        if len(parameter_values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model.parameters, parameter_values))
        return self.model.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor] | BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        """
        Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables
            A sequence of callables (for PyTorch models) or Qiskit operators
            (for quantum circuits). If empty a default observable that
            returns the mean of the network output is used.
        parameter_sets
            Sequence of parameter vectors. Each vector should match the
            dimensionality expected by the model.

        Returns
        -------
        List[List[float]]
            A nested list where the outer dimension indexes the parameter set
            and the inner dimension the observable.
        """
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []

        if self._type == "torch":
            self.model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = _ensure_batch(params)
                    outputs = self.model(inputs)
                    row: List[float] = []
                    for obs in observables:
                        out = obs(outputs)
                        if isinstance(out, torch.Tensor):
                            scalar = float(out.mean().cpu())
                        else:
                            scalar = float(out)
                        row.append(scalar)
                    results.append(row)

        elif self._type == "qiskit":
            for params in parameter_sets:
                state = Statevector.from_instruction(self._bind(params))
                row = [float(state.expectation_value(obs)) for obs in observables]
                results.append(row)

        else:  # callable
            for params in parameter_sets:
                bound = self.model(*params)
                if isinstance(bound, QuantumCircuit):
                    state = Statevector.from_instruction(bound)
                    row = [float(state.expectation_value(obs)) for obs in observables]
                    results.append(row)
                else:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = bound(torch.as_tensor(params, dtype=torch.float32).unsqueeze(0))
                        row = []
                        for obs in observables:
                            out = obs(outputs)
                            if isinstance(out, torch.Tensor):
                                scalar = float(out.mean().cpu())
                            else:
                                scalar = float(out)
                            row.append(scalar)
                        results.append(row)

        if self.shots is not None and self.rng is not None:
            noisy_results = []
            for row in results:
                noisy_results.append(_apply_shot_noise(row, self.shots, self.rng))
            return noisy_results

        return results


__all__ = ["HybridFastEstimator"]
