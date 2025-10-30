"""QuantumNATEnhanced: hybrid classical‑quantum model using PyTorch and torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from math import sqrt


class QuantumSelfAttention(tq.QuantumModule):
    """Quantum block that implements a self‑attention style variational circuit."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.var = tq.Sequential(
            tq.RX(has_params=True, trainable=True),
            tq.RY(has_params=True, trainable=True),
            tq.RZ(has_params=True, trainable=True),
            tq.CRX(has_params=True, trainable=True),
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.random(qdev)
        self.var(qdev)
        return self.measure(qdev)


class ClassicalSelfAttention(nn.Module):
    """Fast CPU‑side self‑attention that mirrors the quantum block."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation: torch.Tensor,
        entangle: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # inputs: (batch, embed_dim)
        query = inputs @ rotation.reshape(self.embed_dim, -1)
        key   = inputs @ entangle.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.t() / sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class QuantumNATEnhanced(nn.Module):
    """Hybrid model that fuses a CNN backbone, a classical self‑attention
    module and a quantum correction circuit."""
    def __init__(self) -> None:
        super().__init__()
        # 1. Convolutional feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Attention parameters – two sets of learnable weights
        self.attn_rot = nn.Parameter(torch.randn(4, 4))
        self.attn_ent = nn.Parameter(torch.randn(4, 4))
        self.attn = ClassicalSelfAttention(embed_dim=4)

        # 3. Quantum sub‑network
        self.qc = QuantumSelfAttention(n_wires=4)

        # 4. Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.batch_norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.backbone(x)
        flat = feat.view(bsz, -1)

        # Classical attention using the learned parameters
        attn_out = self.attn(self.attn_rot, self.attn_ent, flat[:, :4])

        # Quantum path – encode the same 4‑dimensional slice
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device)
        qdev.batch_input = flat[:, :4]
        q_out = self.qc(qdev)

        # Concatenate classical and quantum outputs
        combined = torch.cat([flat, attn_out, q_out], dim=1)
        logits = self.classifier(combined)
        return self.batch_norm(logits)


__all__ = ["QuantumNATEnhanced"]
