"""Hybrid QCNN – quantum implementation with Qiskit.

The quantum network uses a Z‑feature map followed by a stack of
convolutional and pooling layers.  Each convolution block is a
parameterised two‑qubit unitary augmented with a random layer to
inject non‑linear quantum kernels (inspired by quanvolution).  The
circuit is wrapped in an EstimatorQNN for training with classical
optimisers.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals


class HybridQCNN:
    """Quantum hybrid QCNN classifier."""

    def __init__(self, seed: int = 12345) -> None:
        algorithm_globals.random_seed = seed
        self.estimator = StatevectorEstimator()
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self._feature_map.parameters,
            weight_params=self._ansatz.parameters,
            estimator=self.estimator,
        )

    @property
    def _feature_map(self) -> QuantumCircuit:
        return ZFeatureMap(8)

    @property
    def _ansatz(self) -> QuantumCircuit:
        """Build ansatz with convolution, pooling and random layers."""
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolutional block with random kernel
        ansatz.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        # First pooling
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        # Second convolution
        ansatz.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        # Second pooling
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        # Third convolution
        ansatz.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        # Third pooling
        ansatz.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

        return ansatz

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Parameterised two‑qubit convolution unitary."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        # Random layer for quantum‑kernel effect
        qc.compose(self._random_layer(2), [0, 1], inplace=True)
        return qc

    def _random_layer(self, n_qubits: int) -> QuantumCircuit:
        """Random two‑qubit unitary layer."""
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(np.random.rand() * 2 * np.pi, i)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._conv_circuit(params[idx : idx + 3]), [q1, q2])
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(
        self, sources: list[int], sinks: list[int], prefix: str
    ) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks), name="PoolLayer")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, sink, p in zip(sources, sinks, params):
            qc.append(self._pool_circuit(p), [src, sink])
        return qc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the QNN on batch `x`."""
        return self.qnn.predict(x)

    def train(self, X_train, y_train, epochs=10, optimizer=COBYLA()):
        """Placeholder training loop – replace with a full pipeline."""
        self.qnn.fit(X_train, y_train, optimizer=optimizer, epochs=epochs)


__all__ = ["HybridQCNN"]
