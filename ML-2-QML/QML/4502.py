"""Hybrid transformer with quantum modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Quantum kernel utilities
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel implemented with a fixed ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (batch, dim)
        batch = x.size(0)
        self.q_device.reset_states(batch)
        self.encoder(self.q_device, x)
        self.encoder(self.q_device, -y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(-1)

# --------------------------------------------------------------------------- #
# Attention modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class QuantumKernelAttention(MultiHeadAttentionBase):
    """Attention that uses a quantum kernel similarity matrix."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 4) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.kernel = QuantumKernel(n_wires)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, dim = x.size()
        if dim!= self.n_wires:
            raise ValueError("QuantumKernelAttention requires input dim == n_wires")
        kernel_mat = torch.zeros(batch, seq, seq, device=x.device)
        for b in range(batch):
            for i in range(seq):
                for j in range(seq):
                    kernel_mat[b, i, j] = self.kernel(x[b, i, :], x[b, j, :])
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            kernel_mat = kernel_mat.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(kernel_mat, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, x)

# --------------------------------------------------------------------------- #
# Feed‑forward modules
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = QuantumKernelAttention(embed_dim, num_heads, dropout, n_qubits_transformer)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Image feature extractor (NAT style)
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# Hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that mirrors the classical hybrid.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 task: str = "classification",
                 use_image: bool = False,
                 n_qubits_transformer: int = 4,
                 n_qubits_ffn: int = 4) -> None:
        super().__init__()
        self.task = task
        self.use_image = use_image
        self.token_embedding = nn.Embedding(vocab_size, embed_dim) if not use_image else None
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                     n_qubits_transformer, n_qubits_ffn,
                                     dropout=dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if task == "regression":
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.head = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        if use_image:
            self.image_extractor = ConvFeatureExtractor()
            self.image_proj = nn.Linear(4, embed_dim)
        else:
            self.image_extractor = None
            self.image_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_image:
            img_feat = self.image_extractor(x)
            x = self.image_proj(img_feat)
        else:
            tokens = self.token_embedding(x)
            x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.head(x)

__all__ = ["HybridTransformer"]
