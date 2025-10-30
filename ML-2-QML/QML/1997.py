import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class PositionalEmbedding(nn.Module):
    """
    Learnable sinusoidal‑style positional embedding.
    """
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :]
        return x + pe_slice


class FeedForwardClassical(nn.Module):
    """
    Two‑layer perceptron feed‑forward network (fallback for pure classical mode).
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class QuantumProjection(tq.QuantumModule):
    """
    Quantum linear projection that maps a scalar feature vector to the same dimension.
    """
    def __init__(self, dim: int, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.dim = dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(dim)]
        )
        self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(dim)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.q_device = q_device

    def forward(self, x: torch.Tensor, q_device: Optional[tq.QuantumDevice] = None) -> torch.Tensor:
        B, N, D = x.shape
        qdev = q_device or self.q_device or tq.QuantumDevice(n_wires=self.dim, bsz=B * N, device=x.device)
        x_flat = x.view(B * N, D)
        self.encoder(qdev, x_flat)
        for i, gate in enumerate(self.parameters):
            gate(qdev, wires=[i])
        out = self.measure(qdev)
        return out.view(B, N, D)


class QuantumMultiHeadAttention(nn.Module):
    """
    Multi‑head attention where Q, K, V projections are quantum circuits.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_heads = nn.ModuleList([QuantumProjection(self.head_dim, q_device) for _ in range(num_heads)])
        self.k_heads = nn.ModuleList([QuantumProjection(self.head_dim, q_device) for _ in range(num_heads)])
        self.v_heads = nn.ModuleList([QuantumProjection(self.head_dim, q_device) for _ in range(num_heads)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        x_heads = x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q = torch.stack([h(x_heads[:, i]) for i, h in enumerate(self.q_heads)], dim=1)
        k = torch.stack([h(x_heads[:, i]) for i, h in enumerate(self.k_heads)], dim=1)
        v = torch.stack([h(x_heads[:, i]) for i, h in enumerate(self.v_heads)], dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class QuantumFeedForward(nn.Module):
    """
    Feed‑forward network realised by a quantum module followed by classical linear layers.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 q_device: Optional[tq.QuantumDevice] = None, dropout: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qdev = self.q_device.copy(bsz=B * N, device=x.device)
        x_flat = x.view(B * N, D)
        self.encoder(qdev, x_flat)
        for i, gate in enumerate(self.parameters):
            gate(qdev, wires=[i])
        out = self.measure(qdev).view(B, N, self.n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that uses quantum‑augmented attention and feed‑forward layers.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, n_qubits: int = 0,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout, q_device)
        if n_qubits > 0:
            self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits, q_device, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class QuantumHybridTransformer(nn.Module):
    """
    Transformer‑based text classifier with optional quantum submodules.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        max_len: int = 512,
        use_quantum: bool = False,
        n_qubits: int = 0,
        q_device: Optional[tq.QuantumDevice] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(embed_dim, max_len)
        if use_quantum:
            if n_qubits <= 0:
                raise ValueError("n_qubits must be > 0 when use_quantum is True")
            qdev = q_device or tq.QuantumDevice(n_wires=n_qubits)
            self.layers = nn.ModuleList(
                [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                         dropout=dropout, n_qubits=n_qubits,
                                         q_device=qdev) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = ["QuantumHybridTransformer"]
