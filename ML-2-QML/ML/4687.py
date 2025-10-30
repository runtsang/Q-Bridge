"""Hybrid Transformer with optional quantum sub‑modules.

Provides:
- HybridTransformer: a text classifier that can run with classical, quantum,
  or hybrid attention/FFN layers.
- FastEstimator: a lightweight batch estimator that can add shot noise to
  quantum outputs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional, Sequence, Iterable, List, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


# ---- Quantum primitives ---------------------------------------------

class QuantumAttention(tq.QuantumModule):
    """Variational multi‑head attention realised by a parametric circuit."""
    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_wires = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.Parameter(torch.randn(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        q_device.reset()
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            tq.RX(gate, wires=wire)(q_device)
        return self.measure(q_device)


class QuantumFeedForward(tq.QuantumModule):
    """Two‑layer feed‑forward realised by a variational circuit."""
    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_wires = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.Parameter(torch.randn(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, n_qubits * 2)
        self.linear2 = nn.Linear(n_qubits * 2, n_qubits)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        q_device.reset()
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            tq.RY(gate, wires=wire)(q_device)
        out = self.measure(q_device)
        out = self.linear1(out)
        return self.linear2(F.relu(out))


# ---- Hybrid attention & feed‑forward ---------------------------------

class HybridAttention(nn.Module):
    """Mix of classical dot‑product attention and quantum attention."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        quantum_heads: int = 0,
        n_qubits: int = 8,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.quantum_heads = quantum_heads
        self.classical_heads = num_heads - quantum_heads

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

        if quantum_heads:
            # one quantum module that will be split across the quantum heads
            self.q_module = QuantumAttention(n_qubits)

    def forward(self, x: torch.Tensor, q_device: Optional[tq.QuantumDevice] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        # Classical part
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        classical_out = torch.matmul(attn_scores, v)

        # Quantum part (applied to all tokens)
        if self.quantum_heads and q_device is not None:
            qout = self.q_module(x.view(-1, self.embed_dim), q_device)
            qout = qout.view(batch, seq_len, self.embed_dim)
            out = torch.cat([classical_out, qout], dim=-1)
        else:
            out = classical_out

        return self.combine(out)


class HybridFeedForward(nn.Module):
    """Feed‑forward layer with optional quantum sub‑module."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.q_module = QuantumFeedForward(n_qubits)
        else:
            self.linear1 = nn.Linear(embed_dim, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, q_device: Optional[tq.QuantumDevice] = None) -> torch.Tensor:
        if self.use_quantum and q_device is not None:
            return self.q_module(x, q_device)
        else:
            return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---- Transformer block ----------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with optional hybrid attention/FFN."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        quantum_heads: int = 0,
        use_quantum_ffn: bool = False,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = HybridAttention(
            embed_dim, num_heads, dropout, quantum_heads=quantum_heads, n_qubits=n_qubits
        )
        self.ffn = HybridFeedForward(
            embed_dim, ffn_dim, dropout, use_quantum=use_quantum_ffn, n_qubits=n_qubits
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, q_device: Optional[tq.QuantumDevice] = None) -> torch.Tensor:
        attn_out = self.attn(x, q_device)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, q_device)
        return self.norm2(x + self.dropout(ffn_out))


# ---- Positional encoding ---------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---- Hybrid Transformer ----------------------------------------------

class HybridTransformer(nn.Module):
    """Transformer‑based text classifier with optional quantum layers."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        quantum_heads: int = 0,
        use_quantum_ffn: bool = False,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    quantum_heads=quantum_heads,
                    use_quantum_ffn=use_quantum_ffn,
                    n_qubits=n_qubits,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor, q_device: Optional[tq.QuantumDevice] = None) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x, q_device)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# ---- Estimators -----------------------------------------------------

class FastEstimator:
    """Convenience estimator for hybrid models."""
    def __init__(self, model: nn.Module, q_device: Optional[tq.QuantumDevice] = None):
        self.model = model
        self.q_device = q_device

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        results = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                out = self.model(batch, self.q_device)
                row = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    row.append(val)
                results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            results = [
                [float(rng.normal(v, max(1e-6, 1 / shots))) for v in row]
                for row in results
            ]
        return results


# Alias for compatibility with reference pair 2
FastBaseEstimator = FastEstimator

__all__ = [
    "HybridAttention",
    "HybridFeedForward",
    "TransformerBlock",
    "PositionalEncoding",
    "HybridTransformer",
    "FastEstimator",
    "FastBaseEstimator",
]
