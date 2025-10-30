"""Hybrid quantum quanvolution network with quantum self‑attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 image patches."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
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
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (B, 4*14*14)


class QuantumSelfAttention(tq.QuantumModule):
    """Parametric quantum self‑attention block."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation_params = nn.Parameter(
            torch.randn(n_qubits, 3)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(n_qubits - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)

        for i in range(self.n_qubits):
            qdev.rx(self.rotation_params[i, 0], i)
            qdev.ry(self.rotation_params[i, 1], i)
            qdev.rz(self.rotation_params[i, 2], i)

        for i in range(self.n_qubits - 1):
            qdev.crx(self.entangle_params[i], i, i + 1)

        meas = tq.MeasureAll(tq.PauliZ)(qdev)
        return meas  # (B, n_qubits)


class QuanvolutionClassifier(tq.QuantumModule):
    """Quantum quanvolution + quantum self‑attention + classical head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.qattention = QuantumSelfAttention(n_qubits=4)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)            # (B, 4*14*14)
        chunks = torch.chunk(features, 4, dim=1)
        attn_out = []
        for chunk in chunks:
            chunk = chunk.view(x.shape[0], 4)
            attn = self.qattention(chunk)
            attn_out.append(attn)
        attn_features = torch.cat(attn_out, dim=1)  # (B, 4*14*14)
        logits = self.linear(attn_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuantumSelfAttention", "QuanvolutionClassifier"]
