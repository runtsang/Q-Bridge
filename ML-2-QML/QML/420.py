import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np

# --------------------------------------------------------------------------- #
# Quantum transformer primitives – replace classical sub‑modules with PennyLane
# --------------------------------------------------------------------------- #
class QuantumAttention(nn.Module):
    """Parameterised quantum multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, device: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        self.dev = qml.device(device, wires=self.head_dim * num_heads)
        # One rotation per qubit per head
        self.params = nn.Parameter(torch.randn(num_heads, self.head_dim, 3))

    def _qnode(self, vec: torch.Tensor, head_idx: int):
        @qml.qnode(self.dev, interface="torch")
        def circuit(v, p):
            for i in range(self.head_dim):
                qml.RX(v[i], wires=head_idx * self.head_dim + i)
                qml.RY(p[i, 0], wires=head_idx * self.head_dim + i)
                qml.RZ(p[i, 1], wires=head_idx * self.head_dim + i)
            return [qml.expval(qml.PauliZ(w))
                    for w in range(head_idx * self.head_dim,
                                   (head_idx + 1) * self.head_dim)]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.empty(batch, seq, self.embed_dim, device=x.device)
        # Evaluate one token at a time – not efficient but clear
        for b in range(batch):
            for s in range(seq):
                token = x[b, s]
                for h in range(self.num_heads):
                    head_vec = token[h * self.head_dim : (h + 1) * self.head_dim]
                    out[b, s, h * self.head_dim : (h + 1) * self.head_dim] = self._qnode(head_vec, h)(head_vec, self.params[h])
        return out

class QuantumFeedForward(nn.Module):
    """Quantum feed‑forward implemented with a small circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, qubits: int = 8, device: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.qubits = qubits
        self.dev = qml.device(device, wires=qubits)
        self.params = nn.Parameter(torch.randn(qubits, 3))
        self.linear = nn.Linear(qubits, ffn_dim)

    def _qnode(self, vec: torch.Tensor):
        @qml.qnode(self.dev, interface="torch")
        def circuit(v):
            for i in range(self.qubits):
                qml.RX(v[i % vec.shape[0]], wires=i)
                qml.RY(self.params[i, 0], wires=i)
                qml.RZ(self.params[i, 1], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.empty(batch, seq, self.ffn_dim, device=x.device)
        for b in range(batch):
            for s in range(seq):
                token = x[b, s]
                out[b, s] = self.linear(self._qnode(token))
        return out

class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings matching the classical variant."""
    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        seq_len = x.size(1)
        return x + self.pos_emb[:seq_len]

class TextClassifier(nn.Module):
    """Hybrid quantum‑classical transformer classifier."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = LearnedPositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumAttention",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "LearnedPositionalEncoding",
    "TextClassifier",
]
