"""Hybrid QCNN model combining classical convolution emulation and a quantum convolution block."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

# Quantum component imported from the QML module
try:
    from.QCNN__gen051_qml import QCNN_QML
except Exception:
    # Fallback import path for environments without package structure
    from QCNN__gen051_qml import QCNN_QML


class QuantumForward(Function):
    """Autograd wrapper that forwards inputs to a Qiskit EstimatorQNN."""

    @staticmethod
    def forward(ctx, qnn: object, inputs: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy array
        numpy_inputs = inputs.detach().cpu().numpy()
        # EstimatorQNN returns a 2‑D array (n_samples, 1)
        outputs = qnn.predict(numpy_inputs)
        ctx.save_for_backward(inputs)
        return torch.from_numpy(outputs).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        # No gradient through the quantum parameters in this simplified example
        return None, grad_output


class QCNNGenModel(nn.Module):
    """
    Hybrid QCNN consisting of:
    1. Classical feature extractor (linear layers with tanh).
    2. Quantum convolution‑pooling block (via EstimatorQNN).
    3. Classical classifier head.
    """

    def __init__(self, n_features: int = 8, n_qubits: int = 8) -> None:
        super().__init__()
        # Classical front‑end
        self.classical_front = nn.Sequential(
            nn.Linear(n_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
        )
        # Quantum block
        self.qnn = QCNN_QML(num_qubits=n_qubits, feature_dim=n_features)
        self.quantum_forward = QuantumForward.apply
        # Classical back‑end
        self.classical_back = nn.Sequential(
            nn.Linear(1, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classical_front(x)
        q_out = self.quantum_forward(self.qnn, x)
        return torch.sigmoid(self.classical_back(q_out))


def QCNNGen() -> QCNNGenModel:
    """Factory returning the configured QCNNGenModel."""
    return QCNNGenModel()
