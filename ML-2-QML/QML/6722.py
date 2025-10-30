"""Hybrid CNN + QCNN classifier with a variational quantum layer.

The quantum component is built from a QCNN ansatz consisting of
parameterised two‑qubit convolution and pooling blocks.  The
expectation value of a Pauli‑Z observable is used as the quantum
output, which is then passed through a sigmoid to obtain a
probability.  The network is fully differentiable thanks to
EstimatorQNN and can be trained end‑to‑end.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
import qiskit

algorithm_globals.random_seed = 12345

class QCNNQuantumLayer(nn.Module):
    """Quantum layer implementing a QCNN ansatz."""
    def __init__(self, num_qubits: int = 8, shots: int = 100) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)

        # Build the ansatz
        self.ansatz = self._build_ansatz(num_qubits)

        # Estimator and QNN
        estimator = Estimator()
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )

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

    def _conv_layer(self, n_qubits, prefix):
        qc = QuantumCircuit(n_qubits, name="ConvLayer")
        qubits = list(range(n_qubits))
        idx = 0
        params = ParameterVector(prefix, length=n_qubits * 3)
        for a, b in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[idx:idx+3]), [a, b])
            qc.barrier()
            idx += 3
        for a, b in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[idx:idx+3]), [a, b])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, src, sink, prefix):
        n = len(src) + len(sink)
        qc = QuantumCircuit(n, name="PoolLayer")
        idx = 0
        params = ParameterVector(prefix, length=n // 2 * 3)
        for s, t in zip(src, sink):
            qc.append(self._pool_circuit(params[idx:idx+3]), [s, t])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self, n_qubits):
        ansatz = QuantumCircuit(n_qubits, name="QCNNAnsatz")
        # First convolution and pooling
        ansatz.compose(self._conv_layer(n_qubits, "c1"), inplace=True)
        ansatz.compose(self._pool_layer(list(range(n_qubits // 2)),
                                       list(range(n_qubits // 2, n_qubits)), "p1"), inplace=True)
        # Second convolution and pooling
        ansatz.compose(self._conv_layer(n_qubits // 2, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        # Third convolution and pooling
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)
        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)

class HybridQCNNClassifier(nn.Module):
    """Hybrid CNN + QCNN classifier for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout2d(p=0.55)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)  # 8‑dimensional feature vector for QCNN
        self.quantum_layer = QCNNQuantumLayer(num_qubits=8)
        self.out = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 8)
        q_out = self.quantum_layer(x)  # (batch, 1)
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNNClassifier"]
