import math
import numpy as np
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Sequence

class QuantumRBFKernel(tq.QuantumModule):
    """Quantum RBF‑style kernel based on state overlap."""
    def __init__(self, n_wires: int, depth: int = 1, gamma: float = 1.0):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.gamma = gamma
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _encode(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        self.encoder(qdev, x)
        for _ in range(self.depth):
            for wire in range(self.n_wires):
                tqf.cnot(qdev, wires=[wire, (wire + 1) % self.n_wires])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        qdev_x = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0])
        qdev_y = tq.QuantumDevice(n_wires=self.n_wires, bsz=y.shape[0])
        self._encode(qdev_x, x)
        self._encode(qdev_y, y)
        overlap = torch.abs(torch.mean(self.measure(qdev_x) * self.measure(qdev_y)))
        return torch.exp(-self.gamma * overlap**2)

class QuantumAttentionHead(tq.QuantumModule):
    """Encodes a token with a variational circuit and measures."""
    def __init__(self, n_wires: int, depth: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _encode(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        self.encoder(qdev, x)
        for _ in range(self.depth):
            for wire in range(self.n_wires):
                tqf.cnot(qdev, wires=[wire, (wire + 1) % self.n_wires])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self._encode(q_device, x)
        return self.measure(q_device)

class QuantumAttention(tq.QuantumModule):
    """Multi‑head attention where projections are produced by quantum circuits."""
    def __init__(self, embed_dim: int, num_heads: int, depth: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_wires = self.d_k
        self.depth = depth
        self.q_layer = QuantumAttentionHead(self.n_wires, depth)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        proj = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        for i in range(seq_len):
            token = x[:, i, :].reshape(batch_size, self.num_heads, self.d_k)
            for h in range(self.num_heads):
                qdev = self.q_device.copy(bsz=batch_size, device=x.device)
                proj[:, i, :] += self.q_layer(token[:, h, :], qdev)
        return proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_proj = self._project(x)
        attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)
        out, _ = attn(q_proj, q_proj, q_proj)
        return out

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses a quantum‑enhanced attention head."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 depth: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.attn = QuantumAttention(embed_dim, num_heads, depth)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class UnifiedQuantumKernelTransformer(nn.Module):
    """Hybrid model that combines a quantum RBF kernel with a quantum‑enhanced transformer."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 n_wires: int,
                 depth: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.kernel = QuantumRBFKernel(n_wires, depth)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, depth)
              for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        tokens = self.pos_enc(tokens)
        out = self.transformer(tokens)
        out = out.mean(dim=1)
        return self.classifier(out)
