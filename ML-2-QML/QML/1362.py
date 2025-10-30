"""Quantum classifier using a variational circuit with parameter‑shift gradients."""

from __future__ import annotations

import math
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli


class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical interface.
    Provides circuit construction, expectation evaluation, and gradient computation.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend=None,
        shots: int = 1000,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("aer_simulator_statevector")
        self.shots = shots

        # Build circuit and store parameters
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Create a layered ansatz with data encoding and entanglement."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding with RX
        for i, param in enumerate(encoding):
            qc.rx(param, i)

        # Variational layers with Ry and CZ entanglement (ring pattern)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits):
                qc.cz(qubit, (qubit + 1) % self.num_qubits)

        # Observables: single‑qubit Z on each qubit
        observables = [
            SparsePauliOp(Pauli("I" * i + "Z" + "I" * (self.num_qubits - i - 1)))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(weights), observables

    def _run_circuit(self, bound_circuit: QuantumCircuit) -> dict:
        """Execute the bound circuit and return measurement counts."""
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        return result.get_counts()

    def _expectation_from_counts(self, counts: dict, op: SparsePauliOp) -> float:
        """Compute expectation value of a Pauli operator from measurement counts."""
        total = sum(counts.values())
        exp = 0.0
        label = op.to_label()
        for bitstring, cnt in counts.items():
            parity = 1
            for idx, pauli in enumerate(label):
                if pauli == "Z":
                    if bitstring[self.num_qubits - idx - 1] == "1":
                        parity *= -1
            exp += parity * cnt / total
        return exp

    def evaluate(self, data: Iterable[float]) -> List[float]:
        """
        Evaluate expectation values for a single data point.
        Returns a list of expectation values for each observable.
        """
        if len(data)!= self.num_qubits:
            raise ValueError("Input data length must match number of qubits.")
        param_dict = {**dict(zip(self.encoding, data)), **{p: 0.0 for p in self.weights}}
        bound_circuit = self.circuit.bind_parameters(param_dict)
        counts = self._run_circuit(bound_circuit)
        return [self._expectation_from_counts(counts, op) for op in self.observables]

    def parameter_shift_gradient(self, data: Iterable[float]) -> List[float]:
        """
        Compute gradient of the first observable w.r.t. all variational parameters
        using the parameter shift rule.
        Returns a list of gradients corresponding to self.weights.
        """
        shift = math.pi / 2
        base_dict = {**dict(zip(self.encoding, data)), **{p: 0.0 for p in self.weights}}
        base_counts = self._run_circuit(self.circuit.bind_parameters(base_dict))
        base_exp = self._expectation_from_counts(base_counts, self.observables[0])

        gradients = []
        for param in self.weights:
            # +shift
            plus_dict = base_dict.copy()
            plus_dict[param] = shift
            plus_counts = self._run_circuit(self.circuit.bind_parameters(plus_dict))
            exp_plus = self._expectation_from_counts(plus_counts, self.observables[0])

            # -shift
            minus_dict = base_dict.copy()
            minus_dict[param] = -shift
            minus_counts = self._run_circuit(self.circuit.bind_parameters(minus_dict))
            exp_minus = self._expectation_from_counts(minus_counts, self.observables[0])

            grad = 0.5 * (exp_plus - exp_minus)
            gradients.append(grad)
        return gradients

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        backend=None,
        shots: int = 1000,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Static factory that returns the circuit and metadata similar to the ML helper.
        """
        instance = QuantumClassifierModel(num_qubits, depth, backend=backend, shots=shots)
        return instance.circuit, instance.encoding, instance.weights, instance.observables


__all__ = ["QuantumClassifierModel"]
