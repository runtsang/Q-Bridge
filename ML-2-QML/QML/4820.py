"""Quantum hybrid Quanvolution classifier combining quantum filter, self‑attention, and kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumFilter(tq.QuantumModule):
    """Quantum convolutional filter that encodes a 2×2 patch into quantum states."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, patch)
        self.layer(qdev)
        return self.measure(qdev).reshape(-1, self.n_wires)


class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block that applies a random circuit and measures all qubits."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, features: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, features)
        self.layer(qdev)
        return self.measure(qdev).reshape(-1, self.n_qubits)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x)
        self.ansatz(self.q_device, -y)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuanvolutionHybridQuantum(nn.Module):
    """Quantum hybrid network that mirrors the classical architecture."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Quantum convolutional filter
        self.qfilter = QuantumFilter()
        # Quantum self‑attention
        self.attn = QuantumSelfAttention()
        # Quantum kernel
        self.kernel = QuantumKernel()
        # Linear classifier
        self.linear = nn.Linear(4 * 14 * 14 + 1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Split image into non‑overlapping 2×2 patches
        patches = x.view(bsz, 28, 28)
        qdev = tq.QuantumDevice(4, bsz=bsz, device=x.device)

        # 1. Apply quantum filter patch‑wise
        feature_list = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        patches[:, r, c],
                        patches[:, r, c + 1],
                        patches[:, r + 1, c],
                        patches[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                feature_list.append(self.qfilter(qdev, patch))

        features = torch.cat(feature_list, dim=1)  # (B, 4*14*14)

        # 2. Quantum self‑attention over the concatenated features
        attn_out = self.attn(qdev, features)

        # 3. Quantum kernel similarity (self‑kernel for illustration)
        kernel_out = self.kernel(features, features).unsqueeze(-1)

        # 4. Concatenate and classify
        combined = torch.cat([attn_out, kernel_out], dim=-1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridQuantum"]
