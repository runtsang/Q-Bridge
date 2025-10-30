import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(n_qubits, embed_dim)
        self.q_device = qml.device("default.qubit", wires=n_qubits)
        # Initialise learnable parameters for the variational layer
        self.params = nn.Parameter(torch.randn(n_qubits, 3))
        # Create QNode
        self.qnode = qml.QNode(self._q_circuit, self.q_device, interface="torch", diff_method="backprop")

    def _q_circuit(self, inputs, params):
        # Encode inputs into rotation angles
        for i, inp in enumerate(inputs):
            qml.RX(inp, wires=i)
        # Variational layer
        for i in range(self.n_qubits):
            qml.Rot(*params[i], wires=i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.embed_dim)
        x_ffn = self.linear1(x_flat)
        # Map to n_qubits
        if self.ffn_dim >= self.n_qubits:
            q_inputs = x_ffn[:, :self.n_qubits]
        else:
            pad = torch.zeros(x_ffn.shape[0], self.n_qubits - self.ffn_dim, device=x_ffn.device)
            q_inputs = torch.cat([x_ffn, pad], dim=1)
        # Run QNode for each token
        q_outputs = []
        for inp in q_inputs:
            q_outputs.append(self.qnode(inp, self.params))
        q_outputs = torch.stack(q_outputs)  # shape: (batch*seq_len, n_qubits)
        # Apply second linear layer
        out = self.linear2(q_outputs)
        out = out.reshape(batch, seq_len, self.embed_dim)
        return out


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that replaces the feed‑forward with a quantum module."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, n_qubits_ffn: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=n_qubits_ffn, dropout=dropout)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier supporting optional quantum sub‑modules."""
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
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_quantum:
            blocks = [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout, n_qubits_ffn)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTextClassifier",
]
