"""Hybrid quantum model combining CNN, quantum transformer, quantum sampler and kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from typing import Sequence

# Feature extractor (classical part)
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 1, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))

# Positional encoding
class PositionalEncoder(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# Quantum attention block
class QuantumAttention(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.d_k = embed_dim // heads
        self.heads = heads
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.size()
        if dim!= self.d_k * self.heads:
            raise ValueError("Dimension mismatch")
        proj = x.view(batch, seq, self.heads, self.d_k).transpose(1, 2)
        head_outs = []
        for h in range(self.heads):
            token = proj[:, h, :, :].reshape(-1, self.d_k)
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            head_outs.append(out)
        out = torch.stack(head_outs, dim=1).view(batch, seq, -1)
        return self.combine(self.dropout(out))

# Quantum feed‑forward block
class QuantumFeedForward(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            return self.measure(qdev)

    def __init__(self, dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self.QLayer()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for tok in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=tok.size(0), device=tok.device)
            outs.append(self.q_layer(tok, qdev))
        out = torch.stack(outs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# Transformer block
class QuantumTransformerBlock(tq.QuantumModule):
    def __init__(self, dim: int, heads: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = QuantumAttention(dim, heads, dropout)
        self.ffn = QuantumFeedForward(dim, ffn_dim, n_wires, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Quantum sampler head
class QuantumSamplerHead(tq.QuantumModule):
    def __init__(self, dim: int, n_out: int, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.proj = nn.Linear(dim, n_wires)
        self.linear = nn.Linear(n_wires, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = self.q_device.copy(bsz=x.size(0), device=x.device)
        x_proj = self.proj(x)
        self.encoder(qdev, x_proj)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        probs = self.measure(qdev)
        return F.softmax(self.linear(probs), dim=-1)

# Quantum kernel
class QuantumKernel(tq.QuantumModule):
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x)
        for i in range(self.n_wires):
            self.q_device.apply(tq.RY, wires=i, params=-y[:, i])
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Hybrid model
class HybridNATModel(tq.QuantumModule):
    """Quantum hybrid model combining CNN, quantum transformer, quantum sampler and kernel."""
    def __init__(
        self,
        in_ch: int = 1,
        cnn_dim: int = 64,
        embed_dim: int = 64,
        heads: int = 4,
        ffn_dim: int = 128,
        blocks: int = 2,
        n_classes: int = 4,
        n_wires: int = 8,
    ) -> None:
        super().__init__()
        self.extractor = CNNFeatureExtractor(in_ch, cnn_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, heads, ffn_dim, n_wires) for _ in range(blocks)]
        )
        self.sampler = QuantumSamplerHead(embed_dim, n_classes, n_wires)
        self.kernel = QuantumKernel(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)  # [B, C]
        tokens = feats.unsqueeze(1)  # [B, 1, C]
        tokens = self.pos_enc(tokens)
        out = self.transformers(tokens).squeeze(1)
        logits = self.sampler(out)
        return logits

__all__ = ["HybridNATModel"]
