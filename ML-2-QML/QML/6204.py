"""
Quantum classifier that mirrors the classical interface.

Builds a layered ansatz with data encoding, variational parameters,
and an optional fully‑connected sub‑circuit that emulates the
classical FCL layer.  Public API matches the classical version.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend
from typing import Iterable, Tuple, List, Dict

class QuantumClassifierModel:
    """
    Quantum classifier that mirrors the classical interface.
    """
    def __init__(self, num_qubits: int, depth: int,
                 backend: Backend = None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector('x', self.num_qubits)
        weights = ParameterVector('theta', self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Fully‑connected sub‑circuit (single‑qubit rotation)
        fcl_theta = ParameterVector('fcl_theta', 1)
        qc.ry(fcl_theta[0], 0)

        observables = [SparsePauliOp('I' * i + 'Z' + 'I' * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights) + list(fcl_theta), observables

    def run(self,
            data: np.ndarray,
            fcl_theta: float,
            param_binds: Dict = None) -> np.ndarray:
        """
        Execute the circuit with given data and parameter bindings.
        Args:
            data: array of shape (num_qubits + depth*num_qubits,)
                containing values for encoding and variational weights.
            fcl_theta: rotation for the embedded fully‑connected sub‑circuit.
            param_binds: optional dictionary mapping Parameter objects to values.
        Returns:
            expectation value as a 1‑D numpy array.
        """
        if param_binds is None:
            param_binds = {}

        # Bind encoding and variational weights
        all_params = self.encoding + self.weights[:-1]  # exclude fcl_theta
        for param, val in zip(all_params, data):
            param_binds[param] = val

        # Bind the FCL parameter
        param_binds[self.weights[-1]] = fcl_theta  # last weight is fcl_theta

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_binds])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(state, 2) for state in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def build_classifier_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Expose the underlying circuit and metadata for external use.
        """
        return self.circuit, self.encoding, self.weights, self.observables

__all__ = ["QuantumClassifierModel"]
