import math
import time
from typing import Optional

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QAttentionLayer(nn.Module):
    """Quantum attention head implemented with PennyLane."""
    def __init__(self, n_qubits: int, dev: qml.Device):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = dev
        self.weights = nn.Parameter(torch.randn(n_qubits))
        self.qnode = qml.batch(
            qml.qnode(dev, interface="torch", diff_method="backprop")(self._circuit)
        )

    def _circuit(self, x):
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(self.weights[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., n_qubits)
        flat = x.reshape(-1, self.n_qubits)
        out = self.qnode(flat)
        return out.reshape(*x.shape[:-1], self.n_qubits)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each head is a quantum module."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_qubits_per_head: int = 4,
                 dev: qml.Device = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits_per_head = n_qubits_per_head
        self.dev = dev or qml.device("default.qubit", wires=n_qubits_per_head)
        self.attention_heads = nn.ModuleList(
            [QAttentionLayer(n_qubits_per_head, self.dev) for _ in range(num_heads)]
        )
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads
        x = x.view(batch, seq_len, self.num_heads, head_dim)
        out_heads = []
        for i in range(self.num_heads):
            head_input = x[:, :, i, :]  # shape: (batch, seq_len, head_dim)
            if head_dim > self.n_qubits_per_head:
                raise ValueError("head_dim must be <= n_qubits_per_head")
            pad = self.n_qubits_per_head - head_dim
            head_input_padded = F.pad(head_input, (0, pad), "constant", 0)
            flat = head_input_padded.reshape(-1, self.n_qubits_per_head)
            head_out = self.attention_heads[i](flat).reshape(batch, seq_len, self.n_qubits_per_head)
            head_out = head_out[:, :, :head_dim]
            out_heads.append(head_out)
        out = torch.cat(out_heads, dim=-1)
        out = self.combine(out)
        return out


class QFeedForwardLayer(nn.Module):
    """Quantum feed‑forward sub‑module."""
    def __init__(self, n_qubits: int, dev: qml.Device):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = dev
        self.weights = nn.Parameter(torch.randn(n_qubits))
        self.qnode = qml.batch(
            qml.qnode(dev, interface="torch", diff_method="backprop")(self._circuit)
        )

    def _circuit(self, x):
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(self.weights[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.n_qubits)
        out = self.qnode(flat)
        return out.reshape(*x.shape[:-1], self.n_qubits)


class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed‑forward network."""
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits: int = 4,
                 dev: qml.Device = None):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.qff = QFeedForwardLayer(n_qubits, self.dev)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qff(x)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that integrates quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits_per_head: int = 4,
                 n_qubits_ffn: int = 4,
                 dev: qml.Device = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits_per_head, dev)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout,
                                      n_qubits_ffn, dev)

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


class QuantumTransformer(nn.Module):
    """Transformer‑based classifier that uses quantum blocks."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_per_head: int = 4,
                 n_qubits_ffn: int = 4,
                 dev: qml.Device = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout,
                                      n_qubits_per_head, n_qubits_ffn, dev)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.quantum_time = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.classifier(x)
        end = time.perf_counter()
        self.quantum_time += end - start
        return out

    def get_quantum_time(self) -> float:
        """Return the accumulated quantum circuit execution time."""
        return self.quantum_time
