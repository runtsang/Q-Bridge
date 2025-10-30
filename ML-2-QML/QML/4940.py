from __future__ import annotations

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np
from typing import Iterable, List, Sequence

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridSamplerQNN:
    """Quantum implementation of the hybrid sampler.

    Builds a 2‑qubit parameterised circuit with input and weight parameters.
    Provides an evaluate method that uses FastBaseEstimator to compute
    expectation values, and a __call__ method that returns the probability
    distribution over the computational basis.
    """
    def __init__(self) -> None:
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 4)
        self._build_circuit()
        self.estimator = FastBaseEstimator(self.circuit)

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        for w in self.weight_params:
            self.circuit.ry(w, 0 if w.index % 2 == 0 else 1)
        self.circuit.cx(0, 1)

    def __call__(self, parameters: Sequence[float], shots: int = 1024) -> np.ndarray:
        """Return probabilities of measuring 00,01,10,11."""
        bound = self.circuit.assign_parameters(dict(zip(self.input_params.params + self.weight_params.params, parameters)), inplace=False)
        state = Statevector.from_instruction(bound)
        probs = state.probabilities()
        # Order 00,01,10,11
        return np.array(probs, dtype=np.float32)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set."""
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["HybridSamplerQNN"]
