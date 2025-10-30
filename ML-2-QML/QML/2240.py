"""Hybrid Quanvolution model with quantum kernel and fully‑connected head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum implementation of a hybrid quanvolution network.

    The network extracts 2×2 image patches, encodes each patch into a 4‑qubit
    register using Ry rotations, applies a random circuit, and then trainable
    RX/RY layers.  After measuring all qubits the concatenated feature map
    is fed into a linear classifier.  Drop‑out is applied before the head.
    """

    def __init__(self, num_wires: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = num_wires

        # Encoder: Ry per pixel
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random circuit
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(num_wires)))

        # Trainable single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

        # Measurement and linear head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))

        features = torch.cat(patches, dim=1)
        features = self.dropout(features)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
