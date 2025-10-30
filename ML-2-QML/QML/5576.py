"""Hybrid quantum QCNN that augments the classical feature extractor with a variational ansatz and a quantum fully‑connected layer."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNGen483:
    """Quantum implementation of a QCNN that mirrors the classical architecture."""
    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()

        # Feature map
        self.feature_map = ZFeatureMap(8)

        # Build convolutional and pooling layers
        self.ansatz = self._build_ansatz()

        # Final quantum fully‑connected layer
        self.qfc_layer = self._build_qfc_layer()

        # Combine all into one circuit
        self.full_circuit = QuantumCircuit(8)
        self.full_circuit.compose(self.feature_map, range(8), inplace=True)
        self.full_circuit.compose(self.ansatz, range(8), inplace=True)
        # The QFC layer acts on the first four qubits of the ansatz output
        self.full_circuit.compose(self.qfc_layer, range(4), inplace=True)

        # Observable for the 4‑qubit fully‑connected layer
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])

        # EstimatorQNN
        weight_params = [p for p in self.full_circuit.parameters
                         if p not in self.feature_map.parameters]
        self.qnn = EstimatorQNN(
            circuit=self.full_circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=weight_params,
            estimator=self.estimator,
        )

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

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : i * 3 + 3])
            qc.compose(sub, [i, i + 1], inplace=True)
        return qc

    def _pool_layer(self, sources, sinks, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks))
        params = ParameterVector(prefix, length=len(sources) * 3)
        for src, snk, idx in zip(sources, sinks, range(len(sources))):
            sub = self._pool_circuit(params[idx * 3 : idx * 3 + 3])
            qc.compose(sub, [src, snk], inplace=True)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(8)
        qc.compose(self._conv_layer(8, "c1"), range(8), inplace=True)
        qc.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        qc.compose(self._conv_layer(4, "c2"), range(4, 8), inplace=True)
        qc.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        qc.compose(self._conv_layer(2, "c3"), range(6, 8), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return qc

    def _build_qfc_layer(self) -> QuantumCircuit:
        qc = QuantumCircuit(4)
        theta = ParameterVector("qfc", 4)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(theta[0], 0)
        qc.rx(theta[1], 1)
        qc.cx(1, 2)
        qc.rz(theta[2], 2)
        qc.cx(2, 3)
        qc.ry(theta[3], 3)
        return qc

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass using the EstimatorQNN."""
        return self.qnn.predict(inputs)

__all__ = ["QCNNGen483"]
