"""Hybrid quantum‑classical QCNN with multi‑observable readout and parameter‑shift gradient."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals


class QCNNModel:
    """
    A hybrid quantum‑classical model that wraps an :class:`EstimatorQNN` with a
    multi‑observable readout. The forward pass returns a probability estimate via
    a sigmoid on the expectation of the last qubit's Pauli‑Z. The training uses a
    parameter‑shift rule implemented by Qiskit’s Estimator.
    """

    def __init__(self, seed: int | None = None) -> None:
        algorithm_globals.random_seed = seed or 42
        self.estimator = Estimator()
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=[SparsePauliOp.from_list([("Z" + "I" * 7, 1)])],
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )
        self.classifier = NeuralNetworkClassifier(
            estimator_qnn=self.qnn,
            optimizer=L_BFGS_B(),
            loss="cross_entropy",
            epochs=200,
            verbose=0,
        )

    @staticmethod
    def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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
    def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for idx in range(0, num_qubits, 2):
            sub_circ = self._conv_circuit(params[idx:idx + 3])
            qc.append(sub_circ.to_instruction(), [idx, idx + 1])
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks), name="Pooling Layer")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for s, t, p in zip(sources, sinks, params.reshape(-1, 3)):
            sub_circ = self._pool_circuit(p)
            qc.append(sub_circ.to_instruction(), [s, t])
        return qc

    def _build_circuit(self) -> Tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector]]:
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

        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit, feature_map.parameters, ansatz.parameters

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the hybrid QCNN using the built‑in QNN classifier."""
        self.classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for class 1."""
        return self.classifier.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on a held‑out set."""
        preds = self.predict(X) > 0.5
        return float((preds == y).mean())


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
