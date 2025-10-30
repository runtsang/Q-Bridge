"""Quantum estimator that combines an input encoder, a variational layer,
and a state‑vector evaluator.

The class follows the FastBaseEstimator interface of the original
reference but adds a full variational ansatz.  It can be used as a drop‑in
replacement for the classical EstimatorQNN in hybrid training loops.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from qiskit import Aer
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


class FastBaseEstimator:
    """
    Lightweight evaluator for parameterised circuits that returns the
    expectation values of a set of observables for multiple parameter sets.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[SparsePauliOp], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class QuantumEstimator(FastBaseEstimator):
    """
    Parameterised circuit with an input encoder (Ry gates) and a
    variational layer (additional Ry rotations).  The circuit is
    evaluated using a state‑vector simulator.
    """

    def __init__(self, n_qubits: int = 4, backend: Aer = None) -> None:
        self.n_qubits = n_qubits
        circuit = self._build_circuit(n_qubits)
        super().__init__(circuit)
        self.backend = backend or Aer.get_backend("statevector_simulator")

    def _build_circuit(self, n_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        # Encoder: map each input feature to a Ry rotation
        for i in range(n_qubits):
            qc.ry(Parameter(f"x{i}"), i)
        # Variational layer: additional Ry rotations that serve as trainable weights
        for i in range(n_qubits):
            qc.ry(Parameter(f"w{i}"), i)
        return qc

    def run(self, input_vals: Sequence[float], weight_vals: Sequence[float]) -> np.ndarray:
        """
        Evaluate the expectation value of a single Pauli‑Z observable on
        each qubit for the given input and weight parameters.
        """
        if len(input_vals) + len(weight_vals)!= len(self._parameters):
            raise ValueError("Incorrect number of parameters for run().")
        bound = {p: v for p, v in zip(self._parameters[: len(input_vals)], input_vals)}
        bound.update({p: w for p, w in zip(self._parameters[len(input_vals) :], weight_vals)})
        circ = self._circuit.assign_parameters(bound)
        state = Statevector.from_instruction(circ)
        # Observable: Pauli‑Z on each qubit
        obs = [SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])]
        exp = state.expectation_value(obs[0])
        return np.array([exp])

    def evaluate(self, observables: Iterable[SparsePauliOp], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Forward compatibility with the original FastBaseEstimator API.
        """
        return super().evaluate(observables, parameter_sets)


def EstimatorQNN() -> QuantumEstimator:
    """Return a quantum estimator instance."""
    return QuantumEstimator()


__all__ = ["EstimatorQNN", "QuantumEstimator", "FastBaseEstimator"]
