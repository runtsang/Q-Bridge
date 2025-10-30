"""Hybrid estimator that can evaluate either a PyTorch model or a Qiskit circuit.

Features:
- Unified interface for classical and quantum evaluation.
- Supports batch evaluation of multiple scalar observables (PyTorch) or quantum operators.
- Optional Gaussian noise to emulate finite‑shot statistics.
- GPU acceleration for the classical model.
- Automatic device placement and parameter‑count validation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = "qiskit.quantum_info.operators.base_operator.BaseOperator"

class HybridFastEstimator:
    """A unified estimator that works with either a PyTorch model or a Qiskit circuit."""

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        circuit: Optional["qiskit.circuit.QuantumCircuit"] = None,
        backend: Optional["qiskit.providers.BaseBackend"] = None,
        shots: Optional[int] = None,
        device: str | torch.device = "cpu",
        shift: float = np.pi / 2,
    ) -> None:
        if model is None and circuit is None:
            raise ValueError("Either'model' or 'circuit' must be provided.")
        self.model = model
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.device = torch.device(device)

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        if self.circuit is not None:
            if self.backend is None:
                from qiskit import Aer
                self.backend = Aer.get_backend("aer_simulator")
            if self.shots is None:
                self.shots = 1000

    def _ensure_batch(self, values: Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _classical_evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = self._ensure_batch(params)
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

    def _quantum_evaluate(
        self,
        observables: Iterable["qiskit.quantum_info.operators.base_operator.BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from qiskit import transpile, assemble

        if self.circuit is None:
            raise RuntimeError("Quantum circuit not initialized.")
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            if len(values)!= len(self.circuit.parameters):
                raise ValueError("Parameter count mismatch for bound circuit.")
            mapping = dict(zip(self.circuit.parameters, values))
            bound = self.circuit.assign_parameters(mapping, inplace=False)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, "qiskit.quantum_info.operators.base_operator.BaseOperator"]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        noise_shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float | complex]]:
        """Evaluate the estimator for a list of observables and parameter sets.

        Parameters
        ----------
        observables:
            Iterable of scalar functions for the classical model or quantum operators.
        parameter_sets:
            Sequence of parameter vectors; each vector must match the model or circuit
            dimensionality.
        noise_shots:
            If provided, Gaussian noise with variance 1 / shots is added to each result.
        seed:
            Random seed for reproducibility of the noise.
        """
        if self.model is not None:
            raw = self._classical_evaluate(observables, parameter_sets)
        else:
            raw = self._quantum_evaluate(observables, parameter_sets)

        if noise_shots is None:
            return raw

        rng = np.random.default_rng(seed)
        noisy: List[List[float | complex]] = []
        for row in raw:
            noisy_row = [float(rng.normal(float(val), max(1e-6, 1 / noise_shots))) for val in row]
            noisy.append(noisy_row)
        return noisy

    def __repr__(self) -> str:
        if self.model is not None:
            return f"<HybridFastEstimator model={self.model.__class__.__name__} device={self.device}>"
        return f"<HybridFastEstimator circuit={self.circuit.name} shots={self.shots}>"

__all__ = ["HybridFastEstimator"]
