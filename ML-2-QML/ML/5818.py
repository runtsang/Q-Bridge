from __future__ import annotations

import torch
from torch import nn
import numpy as np

# Import the quantum circuit builder from the QML module
# The QML module is expected to expose a function named `build_qcnn_qnn`
# that returns an EstimatorQNN instance.
from.QCNN__gen214_qml import build_qcnn_qnn


class QCNNHybridModel(nn.Module):
    """Hybrid QCNN model combining classical preprocessing and a quantum feature extractor.

    The classical sub‑network mirrors the original QCNNModel, providing a
    lightweight feature map.  The quantum sub‑network is a Qiskit
    EstimatorQNN implementing the three‑layer convolution‑pool architecture.
    The outputs of both branches are summed before a sigmoid activation,
    giving a single scalar prediction.
    """
    def __init__(self) -> None:
        super().__init__()

        # Classical feature extractor – shallow, fully‑connected network
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
        )
        self.head = nn.Linear(4, 1)

        # Quantum neural network – instantiated via the QML helper
        self.qnn = build_qcnn_qnn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical forward pass
        feat = self.feature_map(x)

        # Quantum forward pass – EstimatorQNN expects numpy arrays
        feat_np = feat.detach().cpu().numpy()
        qout_np = self.qnn.predict(feat_np)  # shape (batch, 1)
        qout = torch.tensor(qout_np, dtype=x.dtype, device=x.device)

        # Combine classical and quantum outputs
        combined = feat + qout
        out = torch.sigmoid(self.head(combined))
        return out


def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning the configured hybrid QCNN model."""
    return QCNNHybridModel()


__all__ = ["QCNNHybrid", "QCNNHybridModel"]
