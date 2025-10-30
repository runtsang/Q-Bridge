import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumFeedForward(nn.Module):
    """
    Quantum feed‑forward network implemented with a Pennylane QNode.
    Each token vector is encoded into a quantum circuit, processed
    by a parameterised rotation layer, and measured to produce a
    new representation of the same dimensionality.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = None,
                 dropout: float = 0.1, device: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits or embed_dim
        self.wires = list(range(self.n_qubits))
        self.params = nn.Parameter(torch.randn(self.n_qubits))
        self.qnode = qml.QNode(self._circuit, qml.device(device, wires=self.wires), interface="torch")
        self.linear1 = nn.Linear(self.n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _circuit(self, x, params):
        for i, wire in enumerate(self.wires):
            qml.RX(x[i], wires=wire)
            qml.RY(params[i], wires=wire)
        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out_list = []
        for i in range(batch * seq):
            token = x.view(-1, self.embed_dim)[i]
            if self.embed_dim < self.n_qubits:
                pad = torch.zeros(self.n_qubits - self.embed_dim, device=token.device)
                token = torch.cat([token, pad])
            elif self.embed_dim > self.n_qubits:
                token = token[:self.n_qubits]
            out = self.qnode(token, self.params)
            out_list.append(out)
        out = torch.stack(out_list).reshape(batch, seq, self.n_qubits)
        out = self.linear1(self.dropout(out))
        out = self.linear2(F.relu(out))
        return out


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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


class QuantumTransformerHybrid(nn.Module):
    """
    Transformer that replaces the classical feed‑forward network with a
    quantum one. The architecture otherwise matches the classical
    implementation in the ML module.
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
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.ModuleList(
            [TransformerBlockBase(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.transformers:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumFeedForward",
    "TransformerBlockBase",
    "PositionalEncoder",
    "QuantumTransformerHybrid",
]
