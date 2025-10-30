"""Quantum‑classical quanvolution with multi‑scale patches, variational circuits, attention, and skip‑connection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class MultiScaleQuanvolutionFilter(tq.QuantumModule):
    """Extracts 2×2 and 4×4 patches from a 28×28 image and measures them with a variational circuit."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires_2x2 = 4
        self.n_wires_4x4 = 16

        # Encoder for 2×2 patches: 4 qubits
        self.encoder_2x2 = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires_2x2)
            ]
        )
        # Encoder for 4×4 patches: 16 qubits
        self.encoder_4x4 = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires_4x4)
            ]
        )
        # Variational layers
        self.q_layer_2x2 = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires_2x2)))
        self.q_layer_4x4 = tq.RandomLayer(n_ops=16, wires=list(range(self.n_wires_4x4)))
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        device = x.device
        # Prepare quantum devices
        qdev_2x2 = tq.QuantumDevice(self.n_wires_2x2, bsz=bsz, device=device)
        qdev_4x4 = tq.QuantumDevice(self.n_wires_4x4, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches_2x2 = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # 2×2 patch flattened
                data_2x2 = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                qdev_2x2.reset()
                self.encoder_2x2(qdev_2x2, data_2x2)
                self.q_layer_2x2(qdev_2x2)
                measurement_2x2 = self.measure(qdev_2x2)
                patches_2x2.append(measurement_2x2.view(bsz, -1))

        patches_4x4 = []
        for r in range(0, 28, 4):
            for c in range(0, 28, 4):
                # 4×4 patch flattened
                data_4x4 = torch.stack(
                    [
                        x[:, r + i, c + j]
                        for i in range(4)
                        for j in range(4)
                    ],
                    dim=1,
                )
                qdev_4x4.reset()
                self.encoder_4x4(qdev_4x4, data_4x4)
                self.q_layer_4x4(qdev_4x4)
                measurement_4x4 = self.measure(qdev_4x4)
                patches_4x4.append(measurement_4x4.view(bsz, -1))

        feat_2x2 = torch.cat(patches_2x2, dim=1)  # shape (bsz, 4*14*14)
        feat_4x4 = torch.cat(patches_4x4, dim=1)  # shape (bsz, 16*7*7)
        return torch.cat([feat_2x2, feat_4x4], dim=1)  # shape (bsz, 4*14*14 + 16*7*7)


class Quanvolution__gen221(tq.QuantumModule):
    """Hybrid quantum‑classical model with multi‑scale quanvolution, attention, and skip‑connection."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = MultiScaleQuanvolutionFilter()
        # Dim: 4*14*14 + 16*7*7 = 784 + 784 = 1568
        feature_dim = 4 * 14 * 14 + 16 * 7 * 7
        self.attention = nn.Linear(feature_dim, feature_dim)
        self.skip = nn.Linear(28 * 28, feature_dim)
        self.linear = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        skip_features = self.skip(x.view(x.size(0), -1))
        combined = features + skip_features
        att_weights = torch.sigmoid(self.attention(combined))
        att_features = combined * att_weights
        logits = self.linear(att_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen221"]
