"""
QuantumTransformerQuantum: Fully quantum transformer block built with PennyLane.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, n_qubits_per_head: int,
                 n_layers: int = 2, device: Optional[object] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_qubits = n_qubits_per_head
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits_per_head)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits_per_head))
        def circuit(x, params):
            for l in range(n_layers):
                for q in range(n_qubits_per_head):
                    qml.RY(params[l, q], wires=q)
                for q in range(n_qubits_per_head - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits_per_head - 1, 0])
            for q in range(n_qubits_per_head):
                qml.RY(x[q], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits_per_head)]
        self.qnode = qml.QNode(circuit, self.device, interface="torch")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.size()
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :], -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out_heads = out.view(B, T, self.num_heads, self.head_dim)
        q_heads = []
        for i in range(self.num_heads):
            head = out_heads[:, :, i, :self.n_qubits]
            q_out = self.qnode(head, self.params)
            q_heads.append(q_out)
        out = torch.stack(q_heads, dim=2).reshape(B, T, E)
        return out


class QuantumFeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 n_layers: int = 2, device: Optional[object] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        def circuit(x, params):
            for q in range(n_qubits):
                qml.RY(params[0, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        self.qnode = qml.QNode(circuit, self.device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        out = torch.empty(B, T, self.n_qubits, device=x.device)
        for b in range(B):
            for t in range(T):
                inp = x[b, t, :self.n_qubits]
                out[b, t] = self.qnode(inp, self.params)
        out = self.linear1(out)
        out = self.linear2(F.relu(out))
        return out


class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_per_head: int, n_qubits_ffn: int,
                 n_layers: int = 2, dropout: float = 0.1,
                 device: Optional[object] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, n_qubits_per_head,
                                     n_layers, device=device)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn,
                                      n_layers, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextClassifierQuantum(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, ffn_dim: int, num_classes: int,
                 dropout: float = 0.1, n_qubits_per_head: int = 4,
                 n_qubits_ffn: int = 4, n_layers: int = 2,
                 device: Optional[object] = None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        blocks = [
            TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                    n_qubits_per_head, n_qubits_ffn,
                                    n_layers, dropout, device=device)
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumAttention",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
]
