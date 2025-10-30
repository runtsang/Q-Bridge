"""Hybrid quantum‑classical model that uses a quanvolutional quantum
kernel followed by a classical linear head.

The model combines the encoder and random‑layer construction from
the Quantum‑NAT example with the patch‑based quanvolution filter
from the quanvolution example.  The quantum part outputs a
4‑dimensional feature vector per 2×2 patch; all patch vectors are
concatenated and fed into a classical fully‑connected head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNAT(tq.QuantumModule):
    """Quantum‑classical hybrid model with a quanvolutional kernel."""

    class QLayer(tq.QuantumModule):
        """Random circuit with a few parameterised rotations."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:  # type: ignore[override]
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 4‑dimensional patch to the qubit space
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head identical to the ML version
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device, record_op=True)
        # Reshape image to (B, 28, 28)
        img = x.view(bsz, 28, 28)
        patch_features = []
        # Iterate over 2×2 non‑overlapping patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, r, c],
                        img[:, r, c + 1],
                        img[:, r + 1, c],
                        img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                out = self.measure(qdev)
                patch_features.append(out.view(bsz, 4))
        # Concatenate all patch vectors
        features = torch.cat(patch_features, dim=1)  # shape: (B, 4*14*14)
        logits = self.head(features)
        return self.norm(logits)


__all__ = ["HybridQuantumNAT"]
