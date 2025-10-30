import numpy as np
import torch
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNGenQML:
    """Quantum convolution‑pooling ansatz with a quantum‑kernel head."""

    def __init__(self, n_qubits: int = 8) -> None:
        self.n_qubits = n_qubits
        self.feature_map = ZFeatureMap(n_qubits)
        self.ansatz = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ---------- Convolution & Pooling primitives ----------
    def _conv_circuit(self, params):
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

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 1) * 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = self._pool_circuit(params[idx * 3 : (idx + 1) * 3])
            qc.compose(sub, [s, t], inplace=True)
            qc.barrier()
            idx += 1
        return qc

    # ---------- Build full ansatz ----------
    def _build_ansatz(self):
        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ansatz.compose(
            self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True
        )
        ansatz.compose(self._conv_layer(self.n_qubits // 2, "c2"), inplace=True)
        ansatz.compose(
            self._pool_layer([0, 1], [2, 3], "p2"), inplace=True
        )
        ansatz.compose(self._conv_layer(self.n_qubits // 4, "c3"), inplace=True)
        ansatz.compose(
            self._pool_layer([0], [1], "p3"), inplace=True
        )
        return ansatz

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum neural network."""
        return self.qnn(inputs)


def QCNN() -> QCNNGenQML:
    """Factory returning the configured hybrid QCNN‑style model."""
    return QCNNGenQML()


__all__ = ["QCNN", "QCNNGenQML"]
