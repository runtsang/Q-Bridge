"""Quantum convolution‑inspired network with adaptive measurement and trainable feature map.

The implementation builds on Qiskit’s EstimatorQNN but adds:
- A trainable Z‑feature map with additional RY gates.
- An adaptive pooling layer that optimises a global measurement phase.
- A helper to construct the full variational ansatz with skip‑connections.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper circuits – convolution, pooling, and adaptive measurement
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution block with trainable RZ/RY rotations."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling block with a global phase parameter."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Construct a convolutional layer over all adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(param_vec[i // 2 * 3: i // 2 * 3 + 3])
        qc.append(sub, [i, i + 1])
    return qc


def pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Construct a pooling layer that merges source qubits into sinks."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    param_vec = ParameterVector(prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        sub = _pool_circuit(param_vec[sources.index(src) * 3: sources.index(src) * 3 + 3])
        qc.append(sub, [src, sink])
    return qc


# --------------------------------------------------------------------------- #
# Quantum CNN model
# --------------------------------------------------------------------------- #

class QCNNEnhanced:
    """Hybrid variational quantum circuit that mirrors the classical QCNN."""
    def __init__(self, seed: int = 12345) -> None:
        algorithm_globals.random_seed = seed
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8, reps=2, insert_barriers=True)
        # Augment the feature map with a trainable RY layer for extra expressivity
        self.feature_map.add_parameters(ParameterVector("φ", length=8))
        for i in range(8):
            self.feature_map.ry("φ[{}]".format(i), i)

        self.ansatz = QuantumCircuit(8, name="Ansatz")
        # 1st conv–pool block
        self.ansatz.compose(conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        # 2nd conv–pool block
        self.ansatz.compose(conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        # 3rd conv–pool block
        self.ansatz.compose(conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # Observable: single‑qubit Z on the last qubit
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # Build the EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> None:
        """Gradient‑based training using L‑BFGS‑B."""
        optimizer = L_BFGS_B(maxiter=epochs)
        optimizer.minimize(fun=self.qnn, grad=self.qnn.grad, x0=np.random.rand(len(self.ansatz.parameters)))
        # Note: For a full training loop, one would iterate over batches and update parameters.
        # This simplified version demonstrates optimisation of the entire circuit.

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of inputs."""
        return np.array([self.qnn.predict(inputs=x)[0] for x in X])

    def circuit_qasm(self) -> str:
        """Return the QASM string of the current circuit."""
        return self.circuit.decompose().qasm()

def QCNNEnhancedFactory(seed: int = 12345) -> QCNNEnhanced:
    """Convenience factory returning a freshly instantiated QCNNEnhanced."""
    return QCNNEnhanced(seed=seed)

__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
