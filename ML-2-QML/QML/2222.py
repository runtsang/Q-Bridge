"""QuantumRegressionEnhanced: quantum regression model with transformer layers.

This module implements a quantum‑enhanced regression architecture that
encodes the input state, passes it through a stack of quantum transformer
blocks, and projects the final features to a scalar output.  The design
mirrors the classical implementation but replaces the attention and
feed‑forward sub‑modules with quantum circuits, enabling the model to
learn high‑dimensional quantum features.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchquantum as tq
import torchquantum.functional as tqf

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumEncoder(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)

class QAttention(tq.QuantumModule):
    def __init__(self, num_heads: int, num_wires: int):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([QuantumEncoder(num_wires) for _ in range(num_heads)])
        self.combine = nn.Linear(num_wires * num_heads, num_wires)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        outputs = []
        for head in self.heads:
            outputs.append(head(qdev, x))
        out = torch.cat(outputs, dim=-1)
        return self.combine(out)

class QFeedForward(tq.QuantumModule):
    def __init__(self, num_wires: int, ffn_dim: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumEncoder(num_wires)
        self.linear1 = nn.Linear(num_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, num_wires)
        self.dropout = nn.Dropout(0.1)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        out = self.encoder(qdev, x)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class QuantumTransformerBlock(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_wires: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QAttention(num_heads, num_wires)
        self.ffn = QFeedForward(num_wires, ffn_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        attn_out = self.attn(qdev, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(qdev, x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class QuantumRegressionEnhanced(tq.QuantumModule):
    """
    Quantum regression model that encodes the input state, passes it
    through a stack of quantum transformer blocks, and projects the
    final features to a scalar output.
    """
    def __init__(
        self,
        num_wires: int,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = QuantumEncoder(num_wires)
        self.pos_encoder = PositionalEncoder(num_wires)
        self.transformer = nn.ModuleList(
            [QuantumTransformerBlock(num_wires, num_heads, ffn_dim, num_wires) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(num_wires, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=state_batch.device)
        features = self.encoder(qdev, state_batch)
        features = self.pos_encoder(features)
        for block in self.transformer:
            features = block(qdev, features)
        features = features.mean(dim=1)
        return self.head(self.dropout(features)).squeeze(-1)
