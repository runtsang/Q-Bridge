"""Quantum expectation head for QCNNGen184.

The class `QCNNGen184Quantum` implements an 8‑qubit variational circuit that
mirrors the convolutional structure of the classical counterpart.  It
produces a single expectation value that is then wrapped by the
`Hybrid` layer in the classical network.

The circuit is built with Qiskit and executed on the Aer simulator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNGen184Quantum(EstimatorQNN):
    """EstimatorQNN that implements the 8‑qubit QCNN‑style ansatz."""
    def __init__(self) -> None:
        # Feature map that encodes the input vector
        feature_map = ZFeatureMap(8)
        # Ansatz – a stack of convolution and pooling layers
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolutional layer
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        # First pooling layer
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        # Second convolutional layer
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        # Second pooling layer
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        # Third convolutional layer
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        # Third pooling layer
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)

        # Observable for the expectation value
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        super().__init__(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=Estimator(),
        )

    @staticmethod
    def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        """Build a convolutional sub‑circuit."""
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = QCNNGen184Quantum._single_conv(params[i * 3 : (i + 2) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    @staticmethod
    def _single_conv(params) -> QuantumCircuit:
        """Two‑qubit convolution primitive."""
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
    def _pool_layer(sources, sinks, prefix: str) -> QuantumCircuit:
        """Build a pooling sub‑circuit."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="PoolLayer")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for src, sink in zip(sources, sinks):
            sub = QCNNGen184Quantum._single_pool(params[params.index(src) * 3 : (params.index(src) + 1) * 3])
            qc.append(sub, [src, sink])
        return qc

    @staticmethod
    def _single_pool(params) -> QuantumCircuit:
        """Two‑qubit pooling primitive."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc


__all__ = ["QCNNGen184Quantum"]
