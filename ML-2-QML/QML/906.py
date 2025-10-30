"""Hybrid QCNN with a Qiskit EstimatorQNN followed by a classical classifier head."""

import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class ClassicalHead(nn.Module):
    """Simple feed‑forward head to map quantum expectation values to a scalar output."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QCNNHybrid(nn.Module):
    """Hybrid QCNN: Qiskit EstimatorQNN + classical head."""
    def __init__(self, input_dim: int = 8, num_qubits: int = 8) -> None:
        super().__init__()
        self.qnn = self._build_qnn(input_dim, num_qubits)
        self.head = ClassicalHead(num_qubits)

    def _build_qnn(self, input_dim: int, num_qubits: int) -> EstimatorQNN:
        estimator = Estimator()

        # Feature map: single‑qubit RY rotations parameterised by the input vector
        feature_map = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            feature_map.ry(ParameterVector(f"x{i}", length=1)[0], i)

        # Ansatz: one layer of parameterised two‑qubit gates
        ansatz = QuantumCircuit(num_qubits)
        params = ParameterVector("w", length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            ansatz.rz(params[3 * i], i)
            ansatz.ry(params[3 * i + 1], i + 1)
            ansatz.cx(i, i + 1)
            ansatz.ry(params[3 * i + 2], i + 1)
            ansatz.cx(i + 1, i)

        # Observable for each qubit (Z on each qubit)
        observables = [
            SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
            for _ in range(num_qubits)
        ]

        return EstimatorQNN(
            circuit=ansatz,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum layer outputs a vector of expectation values per qubit
        q_outputs = self.qnn(x)
        return self.head(q_outputs)


def QCNN() -> QCNNHybrid:
    """Factory returning the hybrid QCNN model."""
    return QCNNHybrid()


__all__ = ["QCNN", "QCNNHybrid"]
