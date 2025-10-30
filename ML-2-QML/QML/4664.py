"""Quantum estimator using a 4‑qubit variational circuit.

The circuit encodes two input parameters and one trainable weight
through data re‑uploading and a shallow entangling structure.
An EstimatorQNN() factory returns a qiskit Machine Learning
estimator that can be trained with the standard Qiskit primitives.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np
from collections.abc import Sequence
from typing import Iterable, List

class QuantumEstimator:
    """Parameterised 4‑qubit circuit for regression."""

    def __init__(self) -> None:
        # Parameters: two inputs, one trainable weight
        self.input_params = [Parameter(f"input{i}") for i in range(2)]
        self.weight_param = Parameter("weight")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(4)
        # Data re‑uploading: encode inputs on qubits 0 and 1
        qc.ry(self.input_params[0], 0)
        qc.rx(self.input_params[1], 1)
        # Trainable rotation
        qc.ry(self.weight_param, 2)
        # Entangling layer
        qc.cx(0, 2)
        qc.cx(1, 3)
        # Optional layering
        qc.ry(self.weight_param, 0)
        qc.cx(2, 0)
        qc.h(3)
        return qc

    def observables(self) -> List[Pauli]:
        """Return observable list for regression output."""
        # Use PauliZ on each qubit to obtain 4‑dim output
        return [Pauli("Z" * i + "I" * (3 - i)) for i in range(4)]

    def estimator(self) -> StatevectorEstimator:
        """Return a state‑vector based estimator."""
        return StatevectorEstimator()

def EstimatorQNN() -> QuantumEstimator:
    """Legacy factory returning the quantum estimator."""
    return QuantumEstimator()


class FastBaseEstimator:
    """Fast expectation evaluator for the quantum circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        from qiskit.quantum_info import Statevector
        results: List[List[complex]] = []
        for vals in parameter_sets:
            sv = Statevector.from_instruction(self._bind(vals))
            row = [sv.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_noisy(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, 1 / np.sqrt(shots)) + 1j * rng.normal(val.imag, 1 / np.sqrt(shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "QuantumEstimator",
    "EstimatorQNN",
    "FastBaseEstimator",
]
