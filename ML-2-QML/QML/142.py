"""Extended quantum classifier mirroring the classical interface.

Features
--------
* Configurable entanglement pattern (cz, cx, ry).
* Choice of encoding gate (rx, ry, rz, rzz).
* Supports batch evaluation via Aer statevector simulator.
* Provides methods to retrieve parameters, encoding, and observables.
"""

from __future__ import annotations

from typing import List

import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, PauliSumOp


class QuantumClassifierModel:
    """
    A Qiskit-based variational classifier that exposes the same helper
    attributes as the classical counterpart: ``encoding``, ``weight_sizes``,
    and ``observables``.  The class can evaluate a batch of classical data
    and return expectation values of the observables.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        entangler: str = "cz",
        encoding: str = "rx",
    ):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits / input features.
        depth : int, optional
            Number of variational layers.
        entangler : str, optional
            Entanglement pattern ('cz', 'cx', 'ry').
        encoding : str, optional
            Data‑encoding gate ('rx', 'ry', 'rz', 'rzz').
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.entangler = entangler.lower()
        self.encoding_gate = encoding.lower()
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

        # metadata to mirror the classical interface
        self.weight_sizes: List[int] = [self.weights.numel()]
        self.encoding_params: List[ParameterVector] = [self.encoding]
        self.observables_list: List[SparsePauliOp] = self.observables

    def _build_circuit(self):
        """Construct a layered ansatz with data encoding and entanglement."""
        circuit = QuantumCircuit(self.num_qubits)

        # encoding
        encoding = ParameterVector("x", self.num_qubits)
        for i, q in enumerate(range(self.num_qubits)):
            if self.encoding_gate == "rx":
                circuit.rx(encoding[i], q)
            elif self.encoding_gate == "ry":
                circuit.ry(encoding[i], q)
            elif self.encoding_gate == "rz":
                circuit.rz(encoding[i], q)
            elif self.encoding_gate == "rzz":
                circuit.rzz(encoding[i], q)
            else:
                raise ValueError(f"Unsupported encoding gate: {self.encoding_gate}")

        # variational layers
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                circuit.ry(weights[idx], q)
                idx += 1
            # entanglement
            if self.entangler == "cz":
                for q in range(self.num_qubits - 1):
                    circuit.cz(q, q + 1)
            elif self.entangler == "cx":
                for q in range(self.num_qubits - 1):
                    circuit.cx(q, q + 1)
            elif self.entangler == "ry":
                for q in range(self.num_qubits - 1):
                    circuit.ry(np.pi / 2, q)
                    circuit.cz(q, q + 1)
                    circuit.ry(-np.pi / 2, q)
            else:
                raise ValueError(f"Unsupported entangler: {self.entangler}")

        # observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, encoding, weights, observables

    def get_parameters(self):
        """Return all variational parameters."""
        return list(self.weights)

    def get_encoding(self):
        """Return encoding parameters."""
        return list(self.encoding)

    def get_observables(self):
        """Return observable list."""
        return self.observables

    def evaluate(self, data: np.ndarray, shots: int = 1024):
        """
        Evaluate the circuit on a batch of classical data.

        Parameters
        ----------
        data : np.ndarray, shape (batch_size, num_qubits)
            Classical input values to be encoded.
        shots : int, optional
            Number of shots for expectation estimation (ignored for statevector).

        Returns
        -------
        np.ndarray : shape (batch_size, num_qubits)
            Expectation values of the observables.
        """
        if data.ndim!= 2 or data.shape[1]!= self.num_qubits:
            raise ValueError("Data must have shape (batch_size, num_qubits)")

        backend = Aer.get_backend("statevector_simulator")
        exp_vals = np.zeros((len(data), self.num_qubits))

        for i, x in enumerate(data):
            binding = {self.encoding[j]: float(x[j]) for j in range(self.num_qubits)}
            bound_circ = self.circuit.bind_parameters(binding)

            state = backend.run(bound_circ).result().get_statevector()
            state_fns = CircuitStateFn(state)

            for j, obs in enumerate(self.observables):
                op = PauliSumOp.from_list([(obs, 1.0)])
                expectation = PauliExpectation().convert(StateFn(op) @ state_fns)
                exp_vals[i, j] = expectation.eval().real

        return exp_vals


__all__ = ["QuantumClassifierModel"]
