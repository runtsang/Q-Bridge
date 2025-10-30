"""
ConvGen171 – Quantum implementation using Qiskit variational circuits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable


class ConvGen171Quantum:
    """
    Quantum‑inspired convolution filter that operates on 2‑D image patches.
    For each patch, a parameterized circuit is built where the data values
    determine rotation angles.  The circuit is executed on a qasm simulator
    and the expectation value of Pauli‑Z is returned as the activation.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 500, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend("qasm_simulator")

        # Pre‑build a template circuit; parameters will be bound per patch
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.barrier()
        # Add a shallow random entangling layer
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        self.circuit.measure_all()

    def _bind_and_run(self, data: np.ndarray) -> float:
        """
        Bind the rotation angles to the data values and execute the circuit.
        """
        bind_dict = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data)}
        bound_circuit = self.circuit.bind_parameters(bind_dict)
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Compute expectation of Pauli‑Z: (prob(0)-prob(1)) per qubit
        exp = 0.0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            exp += (self.n_qubits - 2 * ones) * cnt
        exp /= self.shots * self.n_qubits
        return exp

    def run(self, data: np.ndarray) -> float:
        """
        Process a single 2‑D patch and return the quantum activation.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)

        Returns:
            float: expectation value of Pauli‑Z over all qubits.
        """
        flat = data.reshape(-1)
        return self._bind_and_run(flat)


class ConvGen171QuantumClassifier:
    """
    End‑to‑end quantum classifier that stacks the quantum filter with a
    fully‑connected quantum layer.  The quantum layer uses a single‑qubit
    parameterized circuit to produce a scalar output.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 500, threshold: float = 0.5):
        self.filter = ConvGen171Quantum(kernel_size, shots, threshold)
        # Fully‑connected quantum layer
        self.fc = QuantumFullyConnectedLayer(shots=shots)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum filter and fully‑connected layer.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)

        Returns:
            np.ndarray: log‑softmax probabilities over classes.
        """
        qfeat = self.filter.run(data)
        logits = self.fc.run([qfeat])
        return np.log(logits)


class QuantumFullyConnectedLayer:
    """
    Simple parameterized quantum circuit acting as a fully‑connected layer.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 500):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(self.n_qubits))
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for each theta and return the expectation
        of Pauli‑Z as a 1‑D array.
        """
        results = []
        for theta in thetas:
            bound = self.circuit.bind_parameters({self.theta: theta})
            job = execute(bound, self.backend, shots=self.shots)
            counts = job.result().get_counts(bound)
            exp = 0.0
            for bitstring, cnt in counts.items():
                ones = bitstring.count("1")
                exp += (self.n_qubits - 2 * ones) * cnt
            exp /= self.shots * self.n_qubits
            results.append(exp)
        return np.array(results, dtype=np.float32)

__all__ = ["ConvGen171Quantum", "ConvGen171QuantumClassifier", "QuantumFullyConnectedLayer"]
