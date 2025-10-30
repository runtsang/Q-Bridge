"""Hybrid transformer classifier with optional quantum components.

This module mirrors the original QTransformerTorch API but adds the
ability to swap classical and quantum sub‑modules at construction
time.  The public classes are:
- MultiHeadAttentionClassical / MultiHeadAttentionQuantum
- FeedForwardClassical / FeedForwardQuantum
- TransformerBlockClassical / TransformerBlockQuantum
- HybridTextClassifier
- build_classifier_circuit (identical to QuantumClassifierModel.ml)
- build_hybrid_classifier_head (returns either a linear or a quantum EstimatorQNN head)
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Core building blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Quantum‑aware attention alias for API symmetry."""

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Quantum‑aware feed‑forward alias for API symmetry."""

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockClassical):
    """Quantum‑aware transformer block alias for API symmetry."""

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
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
# 2. Helper functions
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a simple multi‑layer feed‑forward network and return metadata."""
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

def build_hybrid_classifier_head(
    embed_dim: int,
    num_classes: int,
    use_quantum: bool = False,
    num_qubits: int = 4,
    depth: int = 2,
) -> nn.Module:
    """Return either a linear head or a quantum EstimatorQNN head."""
    if use_quantum:
        # Lazy import to keep the classical module lightweight
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit.primitives import Estimator
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        def _quantum_circuit() -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
            encoding = ParameterVector("x", num_qubits)
            weights = ParameterVector("theta", num_qubits * depth)
            qc = QuantumCircuit(num_qubits)
            for p, q in zip(encoding, range(num_qubits)):
                qc.rx(p, q)
            idx = 0
            for _ in range(depth):
                for q in range(num_qubits):
                    qc.ry(weights[idx], q)
                    idx += 1
                for q in range(num_qubits - 1):
                    qc.cz(q, q + 1)
            observables = [
                SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
            ]
            return qc, list(encoding), list(weights), observables

        qc, enc, wts, obs = _quantum_circuit()
        estimator = Estimator()
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=obs,
            input_params=enc,
            weight_params=wts,
            estimator=estimator,
        )

        class QuantumClassifierHead(nn.Module):
            def __init__(self, estimator_qnn: EstimatorQNN):
                super().__init__()
                self.estimator_qnn = estimator_qnn

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # The EstimatorQNN expects a 1‑D parameter vector.
                # For demonstration we flatten the batch representation.
                return self.estimator_qnn(x)

        return QuantumClassifierHead(estimator_qnn)
    else:
        return nn.Linear(embed_dim, num_classes)

# --------------------------------------------------------------------------- #
# 3. Hybrid text classifier
# --------------------------------------------------------------------------- #

class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier that can mix classical and quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    transformer_quantum_flags : Optional[Iterable[bool]], optional
        Sequence of booleans indicating whether each transformer block should use the quantum implementation.
        If ``None`` all blocks are classical.
    classifier_use_quantum : bool, optional
        Whether to use a quantum EstimatorQNN head.
    classifier_num_qubits : int, optional
        Number of qubits for the quantum classifier head.
    classifier_depth : int, optional
        Depth of the quantum ansatz.
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
        transformer_quantum_flags: Optional[Iterable[bool]] = None,
        classifier_use_quantum: bool = False,
        classifier_num_qubits: int = 4,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if transformer_quantum_flags is None:
            transformer_quantum_flags = [False] * num_blocks
        if len(transformer_quantum_flags)!= num_blocks:
            raise ValueError("transformer_quantum_flags length must match num_blocks")
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout=dropout)
                if flag
                else TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout=dropout)
                for flag in transformer_quantum_flags
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = build_hybrid_classifier_head(
            embed_dim, num_classes, classifier_use_quantum, classifier_num_qubits, classifier_depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTextClassifier",
    "build_classifier_circuit",
    "build_hybrid_classifier_head",
]
