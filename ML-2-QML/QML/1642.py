"""Quantum QCNN implementation with adaptive feature maps and parameter‑shift gradients."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import List, Optional

class SharedQCNN:
    """
    Quantum convolutional neural network building block.

    Enhancements over the seed:
    * Parameter‑shift rule for analytical gradients.
    * Flexible backend selection (Aer, Qiskit Runtime, etc.).
    * Automatic construction of convolution and pooling layers.
    """
    def __init__(
        self,
        num_qubits: int = 8,
        backend: Optional[str] = None,
        feature_map: Optional[ZFeatureMap] = None,
        seed: Optional[int] = 12345,
    ) -> None:
        self.num_qubits = num_qubits
        self.backend = backend
        self.feature_map = feature_map or ZFeatureMap(num_qubits)
        self.estimator = Estimator(backend=self.backend)
        self.circuit = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)]),
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    # ---------- Convolution / Pooling primitives ----------
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
        qc = QuantumCircuit(num_qubits, name="conv_layer")
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 1) * 3])
            qc.append(sub, [i, i + 1])
        return qc

    def _pool_layer(self, sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources) + len(sinks), name="pool_layer")
        params = ParameterVector(prefix, length=len(sources) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[idx : idx + 3])
            qc.append(sub, [src, sink])
            idx += 3
        return qc

    # ---------- Assemble the ansatz ----------
    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # 1st convolution
        qc.compose(self._conv_layer(self.num_qubits, "c1"), inplace=True)
        # 1st pooling
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 2)),
                list(range(self.num_qubits // 2, self.num_qubits)),
                "p1",
            ),
            inplace=True,
        )
        # 2nd convolution
        qc.compose(self._conv_layer(self.num_qubits // 2, "c2"), inplace=True)
        # 2nd pooling
        qc.compose(
            self._pool_layer(
                list(range(self.num_qubits // 4)),
                list(range(self.num_qubits // 4, self.num_qubits // 2)),
                "p2",
            ),
            inplace=True,
        )
        # 3rd convolution
        qc.compose(self._conv_layer(self.num_qubits // 4, "c3"), inplace=True)
        # 3rd pooling
        qc.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return qc

    # ---------- Public API ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return expectation values for the input feature vectors."""
        return self.qnn.predict(X)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """Compute analytical gradients via parameter‑shift."""
        return self.qnn.gradient(X)

def QCNN(**kwargs) -> SharedQCNN:
    """
    Factory returning a configured SharedQCNN instance.
    """
    return SharedQCNN(**kwargs)

__all__ = ["QCNN", "SharedQCNN"]
