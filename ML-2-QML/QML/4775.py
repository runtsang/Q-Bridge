"""Quantum‑centric hybrid model mirroring the classical Composite model."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend
from typing import Iterable

# ------------------------------------------------------------------
# Quantum convolutional filter (adapted from Conv.py)
# ------------------------------------------------------------------
class ConvCircuit:
    """
    Variational circuit that emulates a convolutional filter.
    Parameters
    ----------
    kernel_size : int
    backend : Backend
    shots : int
    threshold : float
    """
    def __init__(self, kernel_size: int, backend: Backend, shots: int = 100, threshold: float = 127.0):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.barrier()
        from qiskit.circuit.random import random_circuit
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.
        Args:
            data: 2D array with shape (kernel_size, kernel_size).
        Returns:
            float: average probability of measuring |1> across qubits.
        """
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(row)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts()
        counts = 0
        for bitstring, freq in result.items():
            ones = bitstring.count('1')
            counts += ones * freq
        return counts / (self.shots * self.n_qubits)


# ------------------------------------------------------------------
# Quantum fully‑connected layer (adapted from FCL.py)
# ------------------------------------------------------------------
class FullyConnectedCircuit:
    """
    1‑qubit parameterised circuit representing a fully‑connected layer.
    """
    def __init__(self, n_qubits: int, backend: Backend, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        params = [{self.theta: t} for t in thetas]
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=params)
        result = job.result().get_counts()
        counts = 0
        for bitstring, freq in result.items():
            ones = bitstring.count('1')
            counts += ones * freq
        probs = counts / (self.shots * self.n_qubits)
        return np.array([probs])


# ------------------------------------------------------------------
# Quantum classifier circuit (adapted from QuantumClassifierModel.py)
# ------------------------------------------------------------------
class ClassifierCircuit:
    """
    Layered ansatz with explicit encoding and variational parameters.
    """
    def __init__(self, num_qubits: int, depth: int, backend: Backend, shots: int = 100):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = QuantumCircuit(num_qubits)
        for i, param in enumerate(self.encoding):
            self.circuit.rx(param, i)
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                self.circuit.ry(self.weights[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                self.circuit.cz(q, q + 1)
        self.circuit.measure_all()
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    def run(self, params: Iterable[float]) -> np.ndarray:
        """
        Run the classifier circuit with the given encoding parameters.
        Returns the probability of measuring |1> for each qubit.
        """
        if len(params)!= self.num_qubits:
            raise ValueError("Parameter vector size must match number of qubits")
        bind = {self.encoding[i]: params[i] for i in range(self.num_qubits)}
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts()
        probs = []
        for i in range(self.num_qubits):
            ones = 0
            total = 0
            for bits, freq in result.items():
                # bit order: qubit 0 is least significant bit
                if bits[-(i+1)] == '1':
                    ones += freq
                total += freq
            probs.append(ones / total)
        return np.array(probs)


# ------------------------------------------------------------------
# Composite quantum model
# ------------------------------------------------------------------
class HybridQuantumConvClassifier:
    """
    Quantum analogue of HybridConvClassifier.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 127.0,
        fc_qubits: int = 1,
        classifier_depth: int = 2,
        shots: int = 100,
    ):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = ConvCircuit(kernel_size, self.backend, shots, conv_threshold)
        self.fc = FullyConnectedCircuit(fc_qubits, self.backend, shots)
        self.classifier = ClassifierCircuit(fc_qubits, classifier_depth, self.backend, shots)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the full quantum pipeline and return a probability distribution
        over the two classes.
        """
        conv_out = self.conv.run(data)          # scalar
        fc_out = self.fc.run([conv_out])        # array shape (1,)
        cls_out = self.classifier.run(fc_out)   # array shape (n_qubits,)
        # For a 1‑qubit classifier, cls_out[0] is probability of |1>.
        probs = np.array([1 - cls_out[0], cls_out[0]])  # map to class 0/1
        return probs

    def predict(self, data: np.ndarray) -> int:
        probs = self.run(data)
        return int(np.argmax(probs))

__all__ = ["HybridQuantumConvClassifier"]
