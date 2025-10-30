"""HybridQCNet: quantum implementation of the hybrid architecture.

The module builds a quanvolution filter, a QCNN quantum circuit, and a
parameterised hybrid head that outputs a probability distribution for
binary classification.  The design follows the classical counterpart
but replaces each component with a variational quantum circuit
executed on a simulator.
"""

import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


class QuanvCircuit:
    """Quantum emulation of a quanvolution filter."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = self.backend.run(self.circuit, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts()
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class QCNNQuantum:
    """Quantum circuit that implements a simplified QCNN."""
    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.backend = AerSimulator()
        self.estimator = StatevectorEstimator()
        self.circuit = self._build_circuit()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self._input_params(),
            weight_params=self._weight_params(),
            estimator=self.estimator
        )

    def _input_params(self):
        return [Parameter(f"x{i}") for i in range(8)]

    def _weight_params(self):
        return [Parameter(f"w{i}") for i in range(7)]

    def _build_circuit(self):
        qc = QuantumCircuit(8)
        # Feature map: simple Ry rotations on each qubit
        for i in range(8):
            qc.ry(Parameter(f"x{i}"), i)
        # Convolution and pooling layers (simplified)
        qc.cx(0, 1); qc.cx(2, 3); qc.cx(4, 5); qc.cx(6, 7)
        qc.barrier()
        qc.cx(0, 2); qc.cx(1, 3); qc.cx(4, 6); qc.cx(5, 7)
        return qc

    def evaluate(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        param_binds = [
            {p: v for p, v in zip(self._input_params(), inputs)},
            {p: v for p, v in zip(self._weight_params(), weights)}
        ]
        expectation = self.qnn.evaluate(param_binds)[0]
        return expectation


class HybridQuantumHead:
    """Variational quantum circuit that maps a scalar to a probability."""
    def __init__(self, backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def evaluate(self, value: float) -> float:
        bind = {self.theta: value}
        job = self.backend.run(self.circuit, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts()
        p0 = result.get('0', 0) / self.shots
        p1 = result.get('1', 0) / self.shots
        return p0 - p1


class HybridQCNet:
    """Full quantum hybrid network for binary classification."""
    def __init__(self, image_shape, backend=None, shots: int = 1024):
        if backend is None:
            backend = AerSimulator()
        self.backend = backend
        self.shots = shots
        self.quanv = QuanvCircuit(kernel_size=2, backend=backend, shots=shots, threshold=127)
        self.qcnn = QCNNQuantum(shots=shots)
        self.hybrid_head = HybridQuantumHead(backend, shots)

    def run(self, images: np.ndarray) -> np.ndarray:
        """Run the network on a batch of images.

        Args:
            images: np.ndarray of shape (batch, 3, H, W) with pixel values in [0, 255].

        Returns:
            np.ndarray of shape (batch, 2) containing class probabilities.
        """
        probs = []
        for img in images:
            # Flatten image to a 1‑D vector
            flat = img.flatten()
            # Use first 8 values as QCNN inputs
            inputs = flat[:8]
            # Placeholder weights (zero) for inference
            weights = np.zeros(7)
            qcnn_exp = self.qcnn.evaluate(inputs, weights)
            head_exp = self.hybrid_head.evaluate(qcnn_exp)
            probs.append([head_exp, 1 - head_exp])
        return np.array(probs)


__all__ = ["HybridQCNet"]
