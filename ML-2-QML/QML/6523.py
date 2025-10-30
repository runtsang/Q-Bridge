"""Hybrid quantum layer: parameterized quantum FC + quantum self‑attention.

A single Qiskit circuit is constructed that first applies a
parameterized rotation block (acting as a fully connected layer) and
then a small entangling block that implements a quantum analogue of
self‑attention.  The circuit is executed on a simulator and returns
the expectation value of the measured qubit(s).
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from typing import Iterable

class HybridQuantumLayer:
    """Quantum hybrid layer."""

    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Parameterized rotation block (FC)
        self.theta = qiskit.circuit.ParameterVector("theta", self.n_qubits)
        for i, qubit in enumerate(qr):
            circuit.ry(self.theta[i], qubit)

        # Entangling block (self‑attention analog)
        self.entangle = qiskit.circuit.ParameterVector("entangle", self.n_qubits - 1)
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i + 1])

        circuit.measure(qr, cr)
        return circuit

    def run(self, theta_vals: Iterable[float], entangle_vals: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with given parameters.

        Parameters
        ----------
        theta_vals : Iterable[float]
            Rotation angles for the FC block.
        entangle_vals : Iterable[float]
            Entanglement parameters (unused in this simple example but kept
            for API compatibility).

        Returns
        -------
        np.ndarray
            Expectation value of the measured qubits as a single float.
        """
        bind_dict = {self.theta[i]: theta_vals[i] for i in range(self.n_qubits)}
        bind_dict.update({self.entangle[i]: entangle_vals[i] for i in range(self.n_qubits - 1)})

        job = execute(self.circuit.bind_parameters(bind_dict), self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(state, 2) for state in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

def HybridAttentionLayer() -> HybridQuantumLayer:
    """Return a ready‑to‑use quantum hybrid layer."""
    return HybridQuantumLayer()

__all__ = ["HybridAttentionLayer"]
