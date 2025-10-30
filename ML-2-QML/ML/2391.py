"""Hybrid model combining classical CNN, QCNN quantum layer, and final head."""

from __future__ import annotations

import torch
import torch.nn as nn

# Import the quantum QCNN layer from the quantum module
try:
    from.QuantumNAT_qml import QCNNQuantumLayer
except Exception:  # pragma: no cover
    # Fallback for environments where relative import fails
    from QuantumNAT_qml import QCNNQuantumLayer


class HybridQuantumNAT(nn.Module):
    """Hybrid model that merges a classical CNN, a QCNN quantum layer, and a linear head."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to 8 features for QCNN
        self.fc_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )
        # Quantum QCNN layer
        self.qcnn = QCNNQuantumLayer()
        # Final classification head
        self.head = nn.Linear(1, 1)  # QCNN outputs a single expectation value
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        proj = self.fc_proj(flat)
        # QCNN expects (batch, 8)
        qcnn_out = self.qcnn(proj)
        out = self.head(qcnn_out)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
