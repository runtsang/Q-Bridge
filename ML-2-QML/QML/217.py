import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumAttentionHead(nn.Module):
    """Quantum circuit that transforms a vector of size d_k."""
    def __init__(self, d_k: int, n_qubits: int = 8, depth: int = 2):
        super().__init__()
        self.d_k = d_k
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        # Encode each element of the input vector into a rotation
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        for _ in range(self.depth):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.Hadamard(wires=self.n_qubits - 1)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape [d_k]
        return self.qnode(x)

class MultiHeadAttentionQuantum(nn.Module):
    """Attention layer that uses a quantum head for each token."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 depth: int = 2):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.quantum_head = QuantumAttentionHead(self.d_k, n_qubits, depth)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Apply quantum transform to each token
        flat = x.reshape(-1, self.d_k)
        qkv = torch.stack([self.quantum_head(flat[i]) for i in range(flat.shape[0])], dim=0)
        qkv = qkv.reshape(batch, seq, self.d_k)
        q = k = v = qkv
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch, seq, self.embed_dim)
        return self.out_proj(out)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network that applies a quantum transformation."""
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int = 8,
                 depth: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.quantum = QuantumAttentionHead(ffn_dim, n_qubits, depth)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        batch, seq, _ = x.shape
        flat = x.reshape(-1, self.linear1.out_features)
        q_out = torch.stack([self.quantum(flat[i]) for i in range(flat.shape[0])], dim=0)
        q_out = q_out.reshape(batch, seq, self.linear1.out_features)
        return self.linear2(self.dropout(q_out))

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 depth: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits, depth)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, depth)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumTransformer(nn.Module):
    """Quantum transformer using Pennylane for attention and feed‑forward."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 depth: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList([TransformerBlockQuantum(embed_dim,
                                                             num_heads,
                                                             ffn_dim,
                                                             dropout,
                                                             n_qubits,
                                                             depth)
                                    for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumAttentionHead",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoding",
    "QuantumTransformer",
]
