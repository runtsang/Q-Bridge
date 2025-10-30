import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented purely with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        dk = self.embed_dim // self.num_heads
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

class FeedForwardClassical(nn.Module):
    """Simple two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class QGate(nn.Module):
    """
    A learnable quantum gate that produces a scalar scaling factor.
    The gate is a single‑qubit variational circuit with a trainable RX rotation.
    """
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.dev = qml.device("default.qubit", wires=1)

        @qml.qnode(self.dev, interface="torch")
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, batch: int, seq: int) -> torch.Tensor:
        val = self.circuit(self.param)
        gate = torch.sigmoid(val).view(1, 1, 1).expand(batch, seq, 1)
        return gate

class HybridTransformerBlock(nn.Module):
    """
    Quantum‑enhanced transformer block.
    Uses a variational quantum gate to modulate the attention output before the residual addition.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = True,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.qgate = QGate()
        else:
            raise RuntimeError("Quantum gating disabled in the quantum module. Set use_quantum=True.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        if self.use_quantum:
            gate = self.qgate(x.size(0), x.size(1))
            attn_out = attn_out * gate
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))
