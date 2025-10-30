"""Hybrid quanvolution estimator combining classical convolution and quantum feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuantumFeatureExtractor(tq.QuantumModule):
    """
    Implements a quantum kernel that operates on 2×2 image patches.
    For each patch a 4‑qubit state is prepared via Ry rotations,
    a RandomLayer is applied, and all qubits are measured in the Z basis.
    The resulting 4‑bit strings are concatenated to form a feature vector
    that is passed to the classical head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Accepts a flattened feature vector of shape [B, 4*14*14].
        Splits it into 196 patches of 4 qubits each, encodes them, and measures.
        """
        bsz, _ = features.shape
        device = features.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = features.view(bsz, 14, 14, 4)
        out = []
        for r in range(14):
            for c in range(14):
                data = patches[:, r, c, :]
                self.encoder(qdev, data)
                self.q_layer(qdev)
                out.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(out, dim=1)


class HybridQuanvolutionEstimator(nn.Module):
    """
    A hybrid network that first applies a classical 2×2 convolution to reduce spatial resolution,
    then feeds the flattened patches into a quantum module (tq.QuantumModule) which extracts
    non‑linear features via a random quantum circuit.  A linear head maps the quantum
    feature vector to class logits.  The design mirrors the original Quanvolution
    architecture while incorporating an Estimator‑style quantum layer for richer
    representations.
    """

    def __init__(self, in_channels: int = 1, out_features: int = 10) -> None:
        super().__init__()
        # Classical 2×2 stride‑2 convolution
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        # Quantum feature extractor
        self.quantum = QuantumFeatureExtractor()
        # Linear classifier
        self.linear = nn.Linear(4 * 14 * 14, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature map
        features = self.conv(x)          # shape: [B, 4, 14, 14]
        features = features.view(features.size(0), -1)  # flatten
        # Quantum feature extraction
        q_features = self.quantum(features)  # shape: [B, 4*14*14]
        logits = self.linear(q_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionEstimator", "QuantumFeatureExtractor"]
