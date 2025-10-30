"""QCNNHybrid – a parameterised quantum‑classical hybrid network.

The class constructs a variational ansatz comprising multiple
convolution and pooling layers, each built from the same
parameterised sub‑circuits as in the original seed.  It exposes a
single :class:`EstimatorQNN` instance that can be trained with
any classical optimiser.  The design is fully configurable, making
it suitable for benchmarking different layer counts and entanglement
patterns.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Tuple


class QCNNHybrid:
    """Quantum‑classical hybrid QCNN with configurable depth.

    Parameters
    ----------
    num_qubits : int, default 8
        Number of qubits in the circuit.
    conv_layers : int, default 3
        Number of convolution layers.
    pool_layers : int, default 3
        Number of pooling layers.
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        conv_layers: int = 3,
        pool_layers: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.seed = seed
        self.estimator = Estimator()
        self.circuit = self._build_circuit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list(
                [("Z" + "I" * (num_qubits - 1), 1)]
            ),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full QCNN circuit."""
        # Feature map
        self.feature_map = ZFeatureMap(self.num_qubits)

        # Ansatz
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # Add convolution and pooling layers
        for i in range(self.conv_layers):
            ansatz.compose(
                self._conv_layer(self.num_qubits, f"c{i + 1}"),
                range(self.num_qubits),
                inplace=True,
            )
            if i < self.pool_layers:
                ansatz.compose(
                    self._pool_layer(self.num_qubits, f"p{i + 1}"),
                    range(self.num_qubits),
                    inplace=True,
                )

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        return circuit

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Build a convolution layer."""
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for q in range(0, num_qubits, 2):
            sub = self._conv_circuit(
                params[3 * q : 3 * q + 3], q, q + 1
            )
            qc.append(sub, [q, q + 1])
        return qc

    def _conv_circuit(
        self, params: ParameterVector, q1: int, q2: int
    ) -> QuantumCircuit:
        """Parameterised 2‑qubit convolution sub‑circuit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Build a pooling layer."""
        qc = QuantumCircuit(num_qubits, name="PoolLayer")
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for i in range(0, num_qubits - 1, 2):
            sub = self._pool_circuit(
                params[3 * (i // 2) : 3 * (i // 2) + 3], i, i + 1
            )
            qc.append(sub, [i, i + 1])
        return qc

    def _pool_circuit(
        self, params: ParameterVector, q1: int, q2: int
    ) -> QuantumCircuit:
        """Parameterised 2‑qubit pooling sub‑circuit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        return self.qnn.predict(X)

    def get_qnn(self) -> EstimatorQNN:
        """Return the underlying EstimatorQNN object."""
        return self.qnn


def QCNNHybridFactory() -> QCNNHybrid:
    """Return a ready‑to‑use instance of the hybrid QCNN."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
