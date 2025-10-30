"""Hybrid transformer with a Qiskit EstimatorQNN regression head."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------- Classical submodules --------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------- Transformer block --------------------- #
class TransformerBlockHybrid(nn.Module):
    """Hybrid transformer block using classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------- Positional encoder --------------------- #
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
        return x + self.pe[:, :x.size(1)]

# --------------------- Qiskit Estimator wrapper --------------------- #
class QiskitEstimatorWrapper(nn.Module):
    """Wraps a Qiskit EstimatorQNN for regression."""
    def __init__(self) -> None:
        super().__init__()
        # Simple 1‑qubit variational circuit
        param = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param, 0)
        observable = SparsePauliOp.from_list([("Z", 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[param],
            estimator=self.estimator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert tensor to NumPy, run Qiskit estimator, convert back
        np_input = x.detach().cpu().numpy()
        # Qiskit EstimatorQNN expects a 1‑dim vector per example
        # Flatten sequence dimension
        flat = np_input.reshape(np_input.shape[0], -1)
        preds = self.estimator_qnn(flat)
        return torch.tensor(preds, device=x.device, dtype=x.dtype).unsqueeze(-1)

# --------------------- Hybrid Transformer --------------------- #
class HybridTransformerEstimator(nn.Module):
    """Transformer with classical layers and a Qiskit EstimatorQNN regression head."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockHybrid(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.estimator = QiskitEstimatorWrapper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.estimator(x)

__all__ = [
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "QiskitEstimatorWrapper",
    "HybridTransformerEstimator",
]
