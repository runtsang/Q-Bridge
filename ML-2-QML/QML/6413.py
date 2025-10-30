"""Quantum‑enhanced kernel transformer using TorchQuantum."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
#  Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumAnsatz(tq.QuantumModule):
    """Encoder that maps a vector into a quantum state."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.encoder.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.encoder.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Compute overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = QuantumAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
#  Quantum attention
# --------------------------------------------------------------------------- #
class QAttention(nn.Module):
    """Multi‑head attention where projections are produced by a quantum encoder."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layers that map quantum outputs to the transformer space
        self.linear_q = nn.Linear(n_wires, embed_dim, bias=False)
        self.linear_k = nn.Linear(n_wires, embed_dim, bias=False)
        self.linear_v = nn.Linear(n_wires, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum encoder
        self.q_encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.q_params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    def _quantum_encode(self, token: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(token.shape[0])
        self.q_encoder(self.q_device, token)
        for wire, gate in enumerate(self.q_params):
            gate(self.q_device, wires=wire)
        return self.measure(self.q_device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        proj = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), 1, -1)
            proj.append(self._quantum_encode(token))
        proj = torch.stack(proj, dim=1)  # (batch, seq, n_wires)

        q = self.linear_q(proj)
        k = self.linear_k(proj)
        v = self.linear_v(proj)

        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)

# --------------------------------------------------------------------------- #
#  Quantum feed‑forward
# --------------------------------------------------------------------------- #
class QFeedForward(nn.Module):
    """Feed‑forward network realised by a quantum encoder."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_wires = n_wires
        self.q_encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.q_params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

        self.linear1 = nn.Linear(n_wires, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _quantum_encode(self, token: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(token.shape[0])
        self.q_encoder(self.q_device, token)
        for wire, gate in enumerate(self.q_params):
            gate(self.q_device, wires=wire)
        return self.measure(self.q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), 1, -1)
            out.append(self._quantum_encode(token))
        out = torch.stack(out, dim=1)  # (batch, seq, n_wires)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_wires_attn: int = 8, n_wires_ffn: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QAttention(embed_dim, num_heads, dropout, n_wires_attn)
        self.ffn = QFeedForward(embed_dim, ffn_dim, n_wires_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
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
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
#  Hybrid kernel transformer
# --------------------------------------------------------------------------- #
class HybridKernelTransformer(nn.Module):
    """Combines a quantum kernel with a quantum transformer classifier."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        n_wires_kernel: int = 4,
        n_wires_attn: int = 8,
        n_wires_ffn: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel = QuantumKernel(n_wires_kernel)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                      n_wires_attn, n_wires_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_vals = self.kernel(x, x).unsqueeze(-1)  # (batch, batch, 1)
        embed = kernel_vals.mean(dim=1).squeeze(-1).unsqueeze(0)  # (1, batch)
        embed = embed.repeat(self.embed_dim, 1).t()  # (batch, embed_dim)
        embed = embed.unsqueeze(1)  # (batch, 1, embed_dim)
        embed = self.pos_encoder(embed)
        out = self.transformer(embed)
        out = out.mean(dim=1)
        return self.classifier(out)

__all__ = [
    "QuantumAnsatz",
    "QuantumKernel",
    "QAttention",
    "QFeedForward",
    "QuantumTransformerBlock",
    "PositionalEncoder",
    "HybridKernelTransformer",
]
