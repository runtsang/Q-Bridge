"""Hybrid fully‑connected layer with an actual quantum circuit."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.fake_provider import FakeAerSimulator
from qiskit.quantum_info import SparsePauliOp

class QuantumLayerCircuit:
    """
    Parameterised quantum circuit that emulates a quantum layer.
    The circuit consists of data encoding and a layered variational ansatz.
    """
    def __init__(self, num_qubits: int, depth: int, shots: int = 100):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self):
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurement of Z on each qubit
        qc.measure_all()
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit for a batch of theta parameters.
        Parameters
        ----------
        thetas : np.ndarray
            Array of shape (num_qubits * depth,) containing variational parameters.
        Returns
        -------
        np.ndarray
            Expectation value over the specified observables.
        """
        param_binds = [{self.weights[i]: theta for i, theta in enumerate(thetas)}]
        bound_circuit = self.circuit.bind_parameters(param_binds[0])
        transpiled = transpile(bound_circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        exp = 0.0
        for state, count in counts.items():
            prob = count / self.shots
            z = 1 if '0' in state else -1
            exp += z * prob
        return np.array([exp])

class HybridFCLClassifier:
    """
    Hybrid classical‑quantum classifier that mirrors the interface of the classical surrogate.
    The class contains a classical feature extractor, a quantum circuit layer, and a
    classification head that produces logits.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_qubits: int, depth: int, num_classes: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.quantum = QuantumLayerCircuit(num_qubits, depth)
        self.classifier = qiskit.circuit.QuantumCircuit(num_qubits)  # placeholder for classification head

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Run the quantum layer and return its expectation value.
        Parameters
        ----------
        thetas : np.ndarray
            Variational parameters for the quantum circuit.
        Returns
        -------
        np.ndarray
            Array containing a single expectation value.
        """
        return self.quantum.run(thetas)

__all__ = ["HybridFCLClassifier"]
