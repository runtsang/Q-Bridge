"""Quantum implementation of the QCNN with multi‑feature‑map support and optional measurement‑based pooling."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable

__all__ = ["QCNNPlus", "QCNNPlusFactory"]

class QCNNPlus:
    """
    Quantum analogue of the classical QCNNPlus.  Builds a parameterised quantum circuit
    that mirrors the convolution / pooling hierarchy.  Supports three feature‑map
    families (Z, ZZ, Pauli) and an optional measurement‑based pooling strategy.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        feature_map_type: str = "Z",
        pooling_strategy: str = "standard",  # or "measurement"
    ) -> None:
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.pooling_strategy = pooling_strategy

        # Choose feature‑map
        if feature_map_type == "Z":
            self.feature_map = ZFeatureMap(num_qubits)
        elif feature_map_type == "ZZ":
            self.feature_map = ZZFeatureMap(num_qubits)
        elif feature_map_type == "Pauli":
            self.feature_map = PauliFeatureMap(num_qubits)
        else:
            raise ValueError(f"Unsupported feature_map_type: {feature_map_type}")

        # Build ansatz
        self.ansatz = self._build_ansatz()

        # Combine feature map and ansatz into the full circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, range(num_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(num_qubits), inplace=True)

        # Observable used for classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Estimator for the QNN
        self.estimator = Estimator()

        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Unitary used in each convolutional layer."""
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
        """Unitary used in each pooling layer."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Builds a convolutional layer over the given number of qubits."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]),
                [q1, q2],
                inplace=True,
            )
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.compose(
                self._conv_circuit(params[param_index : param_index + 3]),
                [q1, q2],
                inplace=True,
            )
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(
        self,
        sources: Iterable[int],
        sinks: Iterable[int],
        param_prefix: str,
    ) -> QuantumCircuit:
        """Builds a pooling layer that maps qubits in `sources` to `sinks`."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for src, snk in zip(sources, sinks):
            qc.compose(
                self._pool_circuit(params[param_index : param_index + 3]),
                [src, snk],
                inplace=True,
            )
            qc.barrier()
            param_index += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the full ansatz with three conv‑pool stages."""
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # Stage 1
        conv1 = self._conv_layer(self.num_qubits, "c1")
        pool1 = self._pool_layer(list(range(self.num_qubits)), list(range(self.num_qubits)), "p1")
        ansatz.compose(conv1, range(self.num_qubits), inplace=True)
        ansatz.compose(pool1, range(self.num_qubits), inplace=True)

        # Stage 2
        conv2 = self._conv_layer(self.num_qubits // 2, "c2")
        pool2 = self._pool_layer(list(range(self.num_qubits // 2)), list(range(self.num_qubits // 2)), "p2")
        ansatz.compose(conv2, range(self.num_qubits // 2), inplace=True)
        ansatz.compose(pool2, range(self.num_qubits // 2), inplace=True)

        # Stage 3
        conv3 = self._conv_layer(self.num_qubits // 4, "c3")
        pool3 = self._pool_layer([0], [1], "p3")
        ansatz.compose(conv3, range(self.num_qubits // 4), inplace=True)
        ansatz.compose(pool3, range(self.num_qubits // 4), inplace=True)

        if self.pooling_strategy == "measurement":
            ansatz.measure(self.num_qubits - 1, self.num_qubits - 1)
            ansatz.reset(self.num_qubits - 1)
        return ansatz

    def get_qnn(self) -> EstimatorQNN:
        """Return the EstimatorQNN ready for training."""
        return self.qnn

def QCNNPlusFactory(
    num_qubits: int = 8,
    feature_map_type: str = "Z",
    pooling_strategy: str = "standard",
) -> EstimatorQNN:
    """
    Factory that builds a QCNNPlus instance and returns its underlying EstimatorQNN.
    """
    model = QCNNPlus(num_qubits=num_qubits, feature_map_type=feature_map_type, pooling_strategy=pooling_strategy)
    return model.get_qnn()
