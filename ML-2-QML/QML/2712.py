"""Hybrid QCNN – quantum implementation.

This module constructs a variational QCNN circuit that mirrors the
classical depth: convolutional and pooling layers, a Z‑feature map,
and a trainable ansatz.  The circuit is built with Qiskit and
exposed through the same QCNNHybrid class name for API parity.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNHybrid:
    """Quantum QCNN circuit builder.

    The circuit is composed of:
    * A ZFeatureMap acting on 8 qubits.
    * Three convolutional layers (each applying a 2‑qubit unitary).
    * Three pooling layers that reduce the qubit count by half.
    * A variational ansatz with trainable parameters.
    The class exposes a `run` method that returns the expectation value
    of a Z observable on the first qubit, which can be used as a binary
    classifier output.
    """
    def __init__(self, shots: int = 1024) -> None:
        self.shots = shots
        self.estimator = Estimator()
        self.circuit = self._build_circuit()

        # Observables and parameter lists
        self.input_params = self.circuit.parameters
        self.weight_params = list(self.circuit.parameters)  # all parameters are trainable

        # Build the EstimatorQNN once for efficient evaluation
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    @staticmethod
    def _conv_block(params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution unitary."""
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

    @staticmethod
    def _pool_block(params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling unitary."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Apply convolution blocks across adjacent qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            block = self._conv_block(param_vec[i // 2 * 3: i // 2 * 3 + 3])
            qc.append(block, [qubits[i], qubits[i + 1]])
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        """Apply pooling blocks between source and sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        param_vec = ParameterVector(prefix, length=len(sources) * 3)
        for idx, (src, snk) in enumerate(zip(sources, sinks)):
            block = self._pool_block(param_vec[idx * 3: idx * 3 + 3])
            qc.append(block, [src, snk])
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """Assemble the full QCNN circuit."""
        feature_map = ZFeatureMap(8)

        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolution + pooling
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

        # Second convolution + pooling
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)

        # Third convolution + pooling
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit

    def run(self, data: np.ndarray) -> float:
        """Evaluate the QCNN on a single 8‑dimensional sample.

        Args:
            data: 1‑D array of length 8 representing the feature vector.

        Returns:
            float: expectation value of the first‑qubit Z observable.
        """
        if data.ndim!= 1 or data.shape[0]!= 8:
            raise ValueError("Input data must be a 1‑D array of length 8.")
        return float(self.qnn.predict([data])[0])

    def parameters(self) -> list[ParameterVector]:
        """Return all trainable parameters of the circuit."""
        return [self.input_params, self.weight_params]


def QCNN() -> QCNNHybrid:
    """Factory returning a quantum QCNNHybrid instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNN"]
