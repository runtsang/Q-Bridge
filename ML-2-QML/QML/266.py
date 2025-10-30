"""
Quantum QCNN using Qiskit Machine Learning.
It supports variable depth, a learnable ZFeatureMap, and a configurable optimizer with LR scheduling.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.optimizers import COBYLA, Adam
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit_ibm_runtime import IBMRuntimeService, Session
from typing import Iterable, Tuple


class QCNNModel:
    """
    Quantum convolutional neural network wrapper.
    Parameters
    ----------
    feature_map_depth : int
        Number of qubits in the feature map (default 8).
    conv_depth : int
        Number of convolution layers per block (default 3).
    pool_depth : int
        Number of pooling layers per block (default 3).
    optimizer : str
        Optimizer name ('cobyla', 'adam', or'spsa').
    lr : float
        Initial learning rate for Adam.
    """
    def __init__(
        self,
        feature_map_depth: int = 8,
        conv_depth: int = 3,
        pool_depth: int = 3,
        optimizer: str = "adam",
        lr: float = 0.01,
    ) -> None:
        self.feature_map_depth = feature_map_depth
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.optimizer_name = optimizer.lower()
        self.lr = lr
        # Build the quantum circuit once
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (feature_map_depth - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=Estimator(),
        )
        self.classifier = NeuralNetworkClassifier(
            estimator=self.qnn,
            optimizer=self._get_optimizer(),
            training_dataset=None,  # to be set in ``train`` method
            test_dataset=None,
        )

    # ------------------------------------------------------------------ #
    # Circuit construction helpers
    # ------------------------------------------------------------------ #
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 2) * 3])
            qc.append(sub, [i, i + 1])
            qc.barrier()
        return qc

    def _pool_layer(self, sources: Iterable[int], sinks: Iterable[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[:3])
            qc.append(sub, [src, sink])
            qc.barrier()
            params = params[3:]
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        fm = ZFeatureMap(self.feature_map_depth)
        ansatz = QuantumCircuit(self.feature_map_depth, name="Ansatz")

        # First conv & pool block
        ansatz.compose(self._conv_layer(self.feature_map_depth, "c1"), inplace=True)
        ansatz.compose(
            self._pool_layer(list(range(0, self.feature_map_depth, 2)),
                             list(range(1, self.feature_map_depth, 2)),
                             "p1"),
            inplace=True,
        )

        # Subsequent blocks
        for i in range(2, self.conv_depth + 1):
            ansatz.compose(self._conv_layer(self.feature_map_depth // (2 ** (i - 1)), f"c{i}"), inplace=True)
            ansatz.compose(
                self._pool_layer(
                    list(range(0, self.feature_map_depth // (2 ** (i - 1)), 2)),
                    list(range(1, self.feature_map_depth // (2 ** (i - 1)), 2)),
                    f"p{i}",
                ),
                inplace=True,
            )

        # Assemble full circuit
        circuit = QuantumCircuit(self.feature_map_depth)
        circuit.compose(fm, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit

    def _get_optimizer(self):
        if self.optimizer_name == "cobyla":
            return COBYLA()
        elif self.optimizer_name == "adam":
            return Adam(learning_rate=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    # ------------------------------------------------------------------ #
    # Training & evaluation
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_dataset: Tuple[np.ndarray, np.ndarray],
        test_dataset: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        """
        Train the quantum classifier.
        """
        self.classifier.training_dataset = train_dataset
        self.classifier.test_dataset = test_dataset
        self.classifier.fit(epochs=epochs, batch_size=batch_size)

    def evaluate(
        self,
        dataset: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[float, float]:
        """
        Evaluate the trained classifier. Returns loss and accuracy.
        """
        loss, accuracy = self.classifier.score(dataset)
        return loss, accuracy


__all__ = ["QCNNModel"]
