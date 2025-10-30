"""
QCNNQuantumModel: Configurable quantum QCNN with depth control and optimizer choice.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


class QCNNQuantumModel:
    """
    Builds a QCNN with a user‑specified depth.  Each depth adds a
    convolution‑plus‑pooling pair.  The optimizer can be chosen
    between a gradient‑free COBYLA and a stochastic SPSA.
    """
    def __init__(self, depth: int = 3, optimizer: str = "cobyla") -> None:
        self.depth = depth
        self.optimizer_name = optimizer.lower()
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8)
        self.qnn = self._build_qnn()

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Conv")
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx : idx + 3])
            qc.compose(sub, [q1, q2], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pool")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, snk, p in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[p * 3 : p * 3 + 3])
            qc.compose(sub, [src, snk], inplace=True)
            qc.barrier()
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

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8, name="Ansatz")
        # Depth‑controlled stack
        for d in range(self.depth):
            # Convolution
            ansatz.compose(self._conv_layer(8 >> d, f"c{d+1}"), list(range(8 >> d)), inplace=True)
            # Pooling
            sinks = list(range(8 >> d))
            sources = [i for i in range(8 >> d) if i % 2 == 0]
            ansatz.compose(self._pool_layer(sources, sinks, f"p{d+1}"), list(range(8 >> d)), inplace=True)
        return ansatz

    def _build_qnn(self) -> EstimatorQNN:
        ansatz = self._build_ansatz()
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        obs = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        return EstimatorQNN(
            circuit=circuit.decompose(),
            observables=obs,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def get_classifier(self) -> NeuralNetworkClassifier:
        """
        Wraps the EstimatorQNN in a scikit‑learn compatible classifier.
        """
        return NeuralNetworkClassifier(
            qnn=self.qnn,
            optimizer=self._choose_optimizer(),
            epochs=200,
        )

    def _choose_optimizer(self):
        if self.optimizer_name == "cobyla":
            return COBYLA(maxiter=200)
        elif self.optimizer_name == "spsa":
            return SPSA(maxiter=200, perturbation=0.01)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}")

    def train(self, X, y):
        """
        Convenience method to train the QNN on data.
        """
        clf = self.get_classifier()
        clf.fit(X, y)
        return clf


def QCNNQuantum(depth: int = 3, optimizer: str = "cobyla") -> QCNNQuantumModel:
    """
    Factory returning a configurable QCNNQuantumModel.
    """
    return QCNNQuantumModel(depth=depth, optimizer=optimizer)


__all__ = ["QCNNQuantumModel", "QCNNQuantum", "EstimatorQNN"]
