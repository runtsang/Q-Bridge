import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention modules."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced attention that applies a small variational circuit to each head."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False, n_wires: Optional[int] = None):
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.n_wires = n_wires or self.d_k  # one qubit per head dimension
        self.circuit_params = nn.ParameterList([nn.Parameter(torch.randn(self.n_wires)) for _ in range(num_heads)])

    def _quantum_circuit(self, x: torch.Tensor, params: torch.Tensor):
        """Single‑head quantum circuit returning a vector of length n_wires."""
        dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(vec):
            for w in range(self.n_wires):
                qml.RX(vec[w], wires=w)
            for w in range(self.n_wires):
                qml.RZ(params[w], wires=w)
            # entanglement layer
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        return circuit(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        x_heads = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # B, H, T, d_k
        outputs = []
        for h in range(self.num_heads):
            head_vec = x_heads[:, h, :, :]  # B, T, d_k
            flat = head_vec.reshape(-1, self.n_wires)
            out = self._quantum_circuit(flat, self.circuit_params[h])
            out = out.view(B, T, self.n_wires)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # B, H, T, d_k
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.circuit_params = nn.Parameter(torch.randn(n_qubits))
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _quantum_circuit(self, vec: torch.Tensor):
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(v):
            for w in range(self.n_qubits):
                qml.RX(v[w], wires=w)
            for w in range(self.n_qubits):
                qml.RZ(self.circuit_params[w], wires=w)
            for w in range(self.n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)]

        return circuit(vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        outputs = []
        for i in range(B):
            for j in range(T):
                token = x[i, j]
                out = self._quantum_circuit(token)
                outputs.append(out)
        out = torch.stack(outputs, dim=0).view(B, T, self.n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


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


class TransformerBlockClassical(nn.Module):
    """Standard transformer block with classical sub‑components."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int, n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_qubits_transformer)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformer(nn.Module):
    """
    Hybrid transformer that can be instantiated in classical or quantum mode.
    When use_quantum=True, the attention and feed‑forward sub‑modules are replaced
    with their quantum counterparts.  The API mirrors the classical version.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        quantum_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if use_quantum:
            qkw = quantum_kwargs or {}
            n_qubits_transformer = qkw.get("n_qubits_transformer", embed_dim // num_heads)
            n_qubits_ffn = qkw.get("n_qubits_ffn", ffn_dim)
            self.blocks = nn.ModuleList(
                [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn, dropout)
                 for _ in range(num_blocks)]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "PositionalEncoder",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "HybridTransformer",
]
