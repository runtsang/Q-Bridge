"""Quantum QCNN with attention layers.

The class QCNNGen106 builds a variational circuit that implements a
convolution‑pool‑attention pipeline.  Convolution and pooling are
adapted from the original QCNN reference; a new attention sub‑circuit
is added, mirroring the structure of the quantum SelfAttention
example.  The resulting circuit is wrapped in an EstimatorQNN so it
can be used as a quantum neural network.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNGen106:
    """Quantum neural network combining convolution, pooling and attention."""

    def __init__(self) -> None:
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(8)
        self.circuit = self._build_ansatz()

        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            qc.rz(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.cx(i, (i + 1) % num_qubits)
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 2)
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(params[i], i)
            qc.ry(params[i + 1], i + 1)
        return qc

    def _attention_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(num_qubits):
            qc.rx(params[3 * i], i)
            qc.ry(params[3 * i + 1], i)
            qc.rz(params[3 * i + 2], i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        ansatz = QuantumCircuit(8)

        # Feature map
        ansatz.compose(self.feature_map, range(8), inplace=True)

        # First convolution, pooling, attention
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer(8, "p1"), list(range(8)), inplace=True)
        ansatz.compose(self._attention_layer(8, "a1"), list(range(8)), inplace=True)

        # Second convolution, pooling, attention on reduced qubits
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer(4, "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._attention_layer(4, "a2"), list(range(4, 8)), inplace=True)

        # Third convolution, pooling, attention on the smallest block
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer(2, "p3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._attention_layer(2, "a3"), list(range(6, 8)), inplace=True)

        return ansatz

    def run(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Evaluate the quantum network on the given inputs."""
        return self.qnn.predict(inputs, shots=shots)


__all__ = ["QCNNGen106"]
