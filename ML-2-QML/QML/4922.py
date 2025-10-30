"""
Hybrid QCNN – quantum implementation.
Leverages Qiskit to build a variational ansatz that mirrors the classical
convolution & pooling structure.  A feature map encodes the input,
the ansatz applies parameterized gates, and a measurement yields a
single‑qubit expectation that is passed through a classical kernel
and regression head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit.machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import QuantumInstance
from qiskit.providers.fake_provider import FakeStatevectorSimulator

# --------------------------------------------------------------------------- #
#  RBF kernel (same as in the classical module)
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.ansatz(x.view(1, -1), y.view(1, -1)).squeeze()

# --------------------------------------------------------------------------- #
#  Helper functions to build QCNN layers
# --------------------------------------------------------------------------- #
def conv_circuit(params):
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

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(pool_circuit(params[param_index:param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

# --------------------------------------------------------------------------- #
#  Hybrid QCNN – quantum path
# --------------------------------------------------------------------------- #
class QCNNGen113Q(nn.Module):
    """
    Quantum‑enhanced QCNN that mirrors the classical stack:
    - Feature map: ZFeatureMap
    - Ansatz: convolution & pooling layers built from parameterised circuits
    - Measurement: expectation value of Z on the first qubit
    - Post‑processing: classical RBF kernel + regression head
    """
    def __init__(
        self,
        input_dim: int = 8,
        gamma: float = 1.0,
        prototype_count: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        np.random.seed(seed)

        # Feature map
        self.feature_map = ZFeatureMap(input_dim)

        # Build ansatz
        ansatz = QuantumCircuit(input_dim, name="Ansatz")
        ansatz.compose(conv_layer(input_dim, "c1"), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        ansatz.compose(conv_layer(input_dim // 2, "c2"), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        ansatz.compose(conv_layer(input_dim // 4, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Measurement observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (input_dim - 1), 1)])

        # Quantum device & estimator
        backend = FakeStatevectorSimulator()
        estimator = Estimator(quantum_instance=QuantumInstance(backend))
        # Create EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

        # Classical kernel and regression head
        self.kernel = Kernel(gamma)
        self.prototypes = nn.Parameter(
            torch.randn(prototype_count, 1, dtype=torch.float32), requires_grad=False
        )
        self.head = nn.Linear(1 + prototype_count, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Quantum evaluation
        qnn_out = self.qnn(inputs.numpy()).reshape(-1, 1)  # shape (batch, 1)
        qnn_tensor = torch.tensor(qnn_out, dtype=torch.float32, device=inputs.device)

        # Kernel between quantum outputs and prototypes
        kernel_features = torch.cat(
            [
                self.kernel(qnn_tensor[i : i + 1], self.prototypes).squeeze(0)
                for i in range(qnn_tensor.shape[0])
            ],
            dim=0,
        )

        # Combine and regress
        combined = torch.cat([qnn_tensor, kernel_features], dim=1)
        return torch.sigmoid(self.head(combined))

def QCNNGen113Q() -> QCNNGen113Q:
    """Factory returning a ready‑to‑train instance."""
    return QCNNGen113Q()

__all__ = ["KernalAnsatz", "Kernel", "QCNNGen113Q", "QCNNGen113Q"]
