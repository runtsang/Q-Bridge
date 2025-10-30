"""Quantum‑circuit estimator that re‑uses the fast‑classical estimator API.

The estimator accepts a Qiskit QuantumCircuit and provides an ``evaluate``
method that returns expectation values of a list of BaseOperator
observables for a batch of parameter sets.  It also exposes a simple
QuantumSelfAttention block that can be used as a drop‑in replacement
for the classical SelfAttention.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Any

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import Aer, execute


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

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        backend = Aer.get_backend("statevector_simulator")
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class QuantumSelfAttention:
    """Quantum self‑attention block that builds a circuit from parameters
    and returns the measurement probability distribution.
    """

    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params, entangle_params):
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        rotation_params: Any,
        entangle_params: Any,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        probs = {k: v / self.shots for k, v in counts.items()}
        return probs
