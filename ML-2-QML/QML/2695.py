"""HybridTransformer: quantum‑enhanced transformer with fast estimator support."""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastEstimator:
    """Utility to evaluate a quantum‑aware model with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class PositionalEncoding(nn.Module):
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


class QuantumTransformerBlock(nn.Module):
    """Single quantum transformer block implemented with Pennylane."""

    def __init__(self, embed_dim: int, n_wires: int, n_layers: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)
        # trainable parameters
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            for l in range(self.n_layers):
                for w in range(self.n_wires):
                    qml.RY(inputs[w], wires=w)
                for w in range(self.n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[self.n_wires - 1, 0])
                for w in range(self.n_wires):
                    qml.RX(params[l, w, 0], wires=w)
                    qml.RY(params[l, w, 1], wires=w)
                    qml.RZ(params[l, w, 2], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        out = torch.zeros(batch, seq_len, self.n_wires, device=x.device)
        for i in range(seq_len):
            out[:, i, :] = self.circuit(x[:, i, :], self.params)
        return out


class TransformerBlockQuantum(nn.Module):
    """Transformer block that chains a quantum attention and a quantum feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_wires: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.attn = QuantumTransformerBlock(embed_dim, n_wires, n_layers)
        self.ffn = QuantumTransformerBlock(embed_dim, n_wires, n_layers)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# Fallback classical block for optional use
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer with fast estimator support."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = True,
        n_wires: int = 32,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        if use_quantum:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim, n_wires, n_layers
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Convenience wrapper that delegates to FastEstimator."""
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = [
    "QuantumTransformerBlock",
    "TransformerBlockQuantum",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoding",
    "HybridTransformer",
    "FastEstimator",
]
