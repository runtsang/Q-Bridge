"""Quantum-enhanced quanvolution network using a variational circuit and an MLP head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumFilter(nn.Module):
    """Variational quantum filter that processes 2×2 image patches."""
    def __init__(self, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_qubits = 4
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._qcircuit, device=self.device, interface="torch")
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(self.n_qubits * 14 * 14)
        # Residual mapping from 28×28 image to the same feature dimension
        self.res_fc = nn.Linear(1 * 28 * 28, self.n_qubits * 14 * 14)

    def _qcircuit(self, patch: torch.Tensor):
        """Circuit that maps a 4‑element patch to a 4‑dimensional measurement."""
        # Input encoding: rotate each qubit around Y
        for i in range(self.n_qubits):
            qml.RY(patch[i], wires=i)
        # Variational layers
        for _ in range(self.n_layers):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RZ(torch.rand(1).item(), wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = x.view(bsz, 28, 28)
        measurements = []
        # Process all 2×2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r + 2, c:c + 2]          # shape (bsz, 2, 2)
                patch = patch.view(bsz, 4)              # shape (bsz, 4)
                patch_meas = []
                for i in range(bsz):
                    meas = self.qnode(patch[i])         # shape (4,)
                    patch_meas.append(meas)
                patch_meas = torch.stack(patch_meas, dim=0)  # (bsz, 4)
                measurements.append(patch_meas)
        # Concatenate all patch measurements
        out = torch.cat(measurements, dim=1)          # (bsz, 4*14*14)
        # Residual connection
        residual = self.res_fc(x.view(bsz, -1))       # (bsz, 4*14*14)
        out = out + residual
        out = self.bn(out)
        out = self.dropout(out)
        return out

class QuanvolutionNet(nn.Module):
    """Hybrid quanvolution network with a variational quantum filter and an MLP classifier."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuantumFilter()
        self.classifier = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
