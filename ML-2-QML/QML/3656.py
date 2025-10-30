from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridClassifier:
    """
    Quantum counterpart of :class:`HybridClassifier`.
    Builds a layered ansatz that mimics convolution and pooling operations
    on top of a data‑uploading feature map.
    """

    def __init__(self, num_qubits: int, depth: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz()
        self.circuit = self._compose_circuit()
        self.observables = [SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])]
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # First convolution + pooling pair
        qc = self._conv_layer(qc, prefix="c1")
        qc = self._pool_layer(qc, prefix="p1")
        # Additional layers as dictated by depth
        for i in range(2, self.depth + 1):
            qc = self._conv_layer(qc, prefix=f"c{i}")
            qc = self._pool_layer(qc, prefix=f"p{i}")
        return qc

    def _conv_layer(self, qc: QuantumCircuit, prefix: str) -> QuantumCircuit:
        params = ParameterVector(prefix, length=self.num_qubits * 3)
        for i in range(0, self.num_qubits, 2):
            conv = self._conv_circuit(params[3 * (i // 2) : 3 * (i // 2 + 1)])
            qc.append(conv, [i, i + 1])
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

    def _pool_layer(self, qc: QuantumCircuit, prefix: str) -> QuantumCircuit:
        params = ParameterVector(prefix, length=self.num_qubits * 3)
        for i in range(0, self.num_qubits, 2):
            pool = self._pool_circuit(params[3 * (i // 2) : 3 * (i // 2 + 1)])
            qc.append(pool, [i, i + 1])
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

    def _compose_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.feature_map, range(self.num_qubits), inplace=True)
        qc.compose(self.ansatz, range(self.num_qubits), inplace=True)
        return qc

    def get_qnn(self) -> EstimatorQNN:
        return self.qnn

def build_classifier_circuit(num_qubits: int, depth: int, conv_mode: bool = True) -> Tuple[HybridClassifier, Iterable[int], List[int], List[int]]:
    """
    Construct the quantum hybrid classifier and expose the same
    metadata format used by the classical factory.
    """
    model = HybridClassifier(num_qubits, depth)
    encoding = list(range(num_qubits))
    weight_sizes = [p.numel() for p in model.ansatz.parameters()]
    observables = model.observables
    return model, encoding, weight_sizes, observables

def QCNN() -> HybridClassifier:
    """
    Convenience factory that returns a default 8‑qubit quantum CNN
    with three depth layers, matching the original QCNN example.
    """
    return HybridClassifier(num_qubits=8, depth=3)

__all__ = ["HybridClassifier", "build_classifier_circuit", "QCNN"]
