"""Hybrid fully‑connected layer: quantum implementation.

Implements a variational circuit with explicit data‑encoding and a
parameterised ansatz. The circuit can be executed on a Qiskit backend
or simulated locally.  The class exposes a `run` method that
accepts an iterable of feature values and returns a single‑dimensional
numpy array of expectation values, mirroring the classical interface.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, List

class HybridFullyConnectedClassifier:
    """
    Quantum variational classifier with data‑encoding and depth‑controlled ansatz.
    """

    def __init__(self, n_qubits: int = 1, depth: int = 1,
                 backend=None, shots: int = 1000) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend

        # Data‑encoding parameters
        self.encoding = ParameterVector("x", n_qubits)

        # Variational parameters
        self.weights = ParameterVector("theta", n_qubits * depth)

        self.circuit = QuantumCircuit(n_qubits)
        for idx, qubit in enumerate(range(n_qubits)):
            self.circuit.rx(self.encoding[idx], qubit)

        weight_idx = 0
        for _ in range(depth):
            for qubit in range(n_qubits):
                self.circuit.ry(self.weights[weight_idx], qubit)
                weight_idx += 1
            for qubit in range(n_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables: one Z per qubit
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
                            for i in range(n_qubits)]

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit and return expectation values.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of feature values to encode into the circuit.

        Returns
        -------
        np.ndarray
            1‑D array of shape (n_qubits,) containing expectation values.
        """
        param_bindings = [{self.encoding[i]: val for i, val in enumerate(thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_bindings)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array([counts.get(state, 0) for state in
                          sorted(counts, key=lambda s: int(s, 2))]) / self.shots
        states = np.array([int(state, 2) for state in sorted(counts)])
        expectation = np.sum(states * probs)
        return np.array([expectation])

def FCL() -> type[HybridFullyConnectedClassifier]:
    """Return the class that implements the quantum hybrid layer."""
    return HybridFullyConnectedClassifier
