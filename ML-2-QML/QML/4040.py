"""Quantum‑enhanced transformer classifier mirroring the classical API.

This module implements a transformer that can optionally use quantum sub‑modules
for attention and feed‑forward, and replaces the classical sampler with a
QiskitSamplerQNN.  All other components remain identical to the classical
implementation so that the class can be used as a drop‑in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

# Classical components reused from the original implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim // self.num_heads) ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

# Quantum components
class MultiHeadAttentionQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = tq.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = tq.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = tq.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = tq.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(out)

class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = tq.Sequential(
            tq.Linear(embed_dim, ffn_dim),
            tq.ReLU(),
            tq.Dropout(dropout),
            tq.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

def SamplerQNN_quantum() -> QiskitSamplerQNN:
    """Return a QiskitSamplerQNN instance that samples from a 2‑qubit circuit."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = StatevectorSampler()
    return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

class SamplerQNN(nn.Module):
    """Fallback classical sampler used when quantum sampler is not requested."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridTextClassifier(nn.Module):
    """Quantum‑enhanced transformer classifier mirroring the classical API."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        use_quantum_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        if use_quantum_attention or use_quantum_ffn:
            self.transformer = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )

        self.dropout = nn.Dropout(dropout)

        self.classifier_head = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        self.sampler = SamplerQNN_quantum() if use_quantum_sampler else SamplerQNN()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier_head(x)
        probs = self.sampler(logits)
        return logits, probs

__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlockClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "SamplerQNN_quantum",
    "HybridTextClassifier",
]
