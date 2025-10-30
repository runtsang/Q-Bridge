"""Quantum QCNN implementation with adaptive pooling and training utilities."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNGen:
    """Quantum implementation of a convolution–pooling network.

    The circuit is constructed from modular convolutional and pooling layers.
    An adaptive pooling strategy is used: after each pooling layer the state
    of the first qubit is measured as the output.  The model can be trained
    with a COBYLA optimiser and provides ``predict`` and ``fit`` helpers.
    """

    def __init__(self, num_qubits: int = 8, seed: int = 12345) -> None:
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(num_qubits, entanglement="linear", reps=1)
        self.ansatz = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # First convolution + pooling
        qc.compose(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        qc.compose(
            self._pool_layer(list(range(self.num_qubits // 2)), list(range(self.num_qubits // 2, self.num_qubits)), "p1"),
            range(self.num_qubits),
            inplace=True,
        )
        # Second convolution + pooling on reduced qubits
        reduced = self.num_qubits // 2
        qc.compose(self._conv_layer(reduced, "c2"), list(range(reduced)), inplace=True)
        qc.compose(
            self._pool_layer(
                list(range(reduced // 2)), list(range(reduced // 2, reduced)), "p2"
            ),
            list(range(reduced)),
            inplace=True,
        )
        # Third convolution on single qubit
        qc.compose(self._conv_layer(1, "c3"), [0], inplace=True)
        return qc

    def _conv_layer(self, n: int, prefix: str) -> QuantumCircuit:
        """Build a 2‑qubit convolution block for n qubits."""
        qc = QuantumCircuit(n)
        params = ParameterVector(prefix, length=n * 3)
        idx = 0
        for q1 in range(0, n, 2):
            if q1 + 1 >= n:
                break
            block = self._conv_circuit(params[idx : idx + 3])
            qc.append(block, [q1, q1 + 1])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        """Adapts a 2‑qubit pooling block between source and sink qubits."""
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            block = self._pool_circuit(params[idx : idx + 3])
            qc.append(block, [s, t])
            qc.barrier()
            idx += 3
        return qc

    def _conv_circuit(self, params) -> QuantumCircuit:
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

    def _pool_circuit(self, params) -> QuantumCircuit:
        """Two‑qubit pooling primitive."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> None:
        """Train the QCNN using COBYLA."""
        optimizer = COBYLA(maxiter=200, disp=False)
        self.qnn.set_optimizer(optimizer)
        self.qnn.fit(X, y, epochs=epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the model predictions."""
        return self.qnn.predict(X)

    @staticmethod
    def default() -> "QCNNGen":
        """Return a quantum QCNN with default settings."""
        return QCNNGen()


__all__ = ["QCNNGen"]
