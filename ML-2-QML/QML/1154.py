import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def compose_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        q, k, v = self.separate_heads(q), self.separate_heads(k), self.separate_heads(v)
        attn, _ = self.attention(q, k, v, mask)
        return self.compose_heads(self.out_proj(attn))

class QuantumKernelAttention(MultiHeadAttentionBase):
    """Attention that uses a learnable quantum kernel per head."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.n_qubits = n_qubits
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.kernel_params = nn.Parameter(torch.randn(n_qubits, n_qubits))
        self._kernel_qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.device, interface='torch', diff_method='backprop')
        def _qnode(x):
            # x: (batch, n_qubits)
            for i in range(self.device.n_wires):
                qml.RX(x[:, i], wires=i)
            for i in range(self.device.n_wires):
                qml.RZ(self.kernel_params[i, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.device.n_wires)]
        return _qnode

    def _quantum_kernel(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.device.n_wires - x.shape[-1]
        if pad > 0:
            x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=-1)
        return self._kernel_qnode(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        q, k, v = self.separate_heads(q), self.separate_heads(k), self.separate_heads(v)
        batch, heads, seq, d_k = q.shape
        q_flat = q.reshape(-1, d_k)
        k_flat = k.reshape(-1, d_k)
        q_meas = self._quantum_kernel(q_flat)
        k_meas = self._quantum_kernel(k_flat)
        q_meas = q_meas.reshape(batch, heads, seq, self.device.n_wires)
        k_meas = k_meas.reshape(batch, heads, seq, self.device.n_wires)
        sim = torch.einsum('bhqd,bhkd->bhqk', q_meas, k_meas)
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.compose_heads(out)
        return self.out_proj(out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward implemented with a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_qubits))
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self._ffn_qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.device, interface='torch', diff_method='backprop')
        def _qnode(x):
            # x: (batch, n_qubits)
            for i in range(self.device.n_wires):
                qml.RX(x[:, i], wires=i)
            for i in range(self.device.n_wires):
                qml.RZ(self.params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.device.n_wires)]
        return _qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embed_dim = x.shape
        x_flat = x.reshape(-1, embed_dim)
        pad = self.device.n_wires - embed_dim
        if pad > 0:
            x_flat = torch.cat([x_flat, torch.zeros(x_flat.shape[0], pad, device=x_flat.device)], dim=1)
        out = self._ffn_qnode(x_flat)
        out = out.reshape(batch, seq, self.device.n_wires)
        out = self.linear1(self.dropout(out))
        out = self.linear2(F.relu(out))
        return out

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum_attention: bool = True,
                 use_quantum_ffn: bool = True,
                 n_qubits_ffn: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = QuantumKernelAttention(embed_dim, num_heads, dropout) if use_quantum_attention else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=n_qubits_ffn, dropout=dropout) if use_quantum_ffn else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000, learned: bool = False):
        super().__init__()
        self.learned = learned
        if learned:
            self.pe = nn.Embedding(max_len, embed_dim)
        else:
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
            pe = torch.zeros(max_len, embed_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            return x + self.pe(positions)
        else:
            return x + self.pe[:, :x.size(1)]

class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 n_qubits_ffn: int = 8,
                 learned_positional: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim, learned=learned_positional)
        blocks = [
            TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                   use_quantum_attention=use_quantum_attention,
                                   use_quantum_ffn=use_quantum_ffn,
                                   n_qubits_ffn=n_qubits_ffn,
                                   dropout=dropout)
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "QuantumKernelAttention",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
