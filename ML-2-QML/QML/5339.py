from __future__ import annotations

import math
from typing import Optional, Callable, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ------------------------------------
# Quantum fast estimator
# ------------------------------------
class QuantumFastBaseEstimator:
    """Evaluate expectation values of a parameterised Qiskit circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ------------------------------------
# Quantum self‑attention
# ------------------------------------
class QuantumSelfAttention(MultiHeadAttentionBase):
    """Variational self‑attention implemented with TorchQuantum."""
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int = 8):
        super().__init__(embed_dim, num_heads)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.n_wires = n_wires

    def _apply_quantum(self, proj: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, proj)
        return self.measure(q_device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=x.device)
        q_enc = self._apply_quantum(q, qdev)
        k_enc = self._apply_quantum(k, qdev)
        v_enc = self._apply_quantum(v, qdev)
        attn_out, _ = self.attention(q_enc, k_enc, v_enc, mask)
        return attn_out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

# ------------------------------------
# Quantum quanvolution
# ------------------------------------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "ry", "wires": [idx]} for idx in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        self.random_layer(q_device)
        return self.measure(q_device)

class QuantumQuanvolutionClassifier(nn.Module):
    """Hybrid network: quantum filter + linear head."""
    def __init__(self, n_wires: int = 4, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(n_wires)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.qfilter.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                out = self.qfilter(patch, qdev)
                patches.append(out.view(bsz, 4))
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# ------------------------------------
# Quantum transformer components
# ------------------------------------
class MultiHeadAttentionBase(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
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

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _apply_quantum(self, proj: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, proj)
        return self.measure(q_device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=x.device)
        q_enc = self._apply_quantum(q, qdev)
        k_enc = self._apply_quantum(k, qdev)
        v_enc = self._apply_quantum(v, qdev)
        attn_out, _ = self.attention(q_enc, k_enc, v_enc, mask)
        return attn_out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

class FeedForwardBase(tq.QuantumModule):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=x.device)
        out = self.encoder(qdev, x)
        out = self.measure(qdev)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_wires_transformer: int,
        n_wires_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires_transformer)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(tq.QuantumModule):
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

class QTransformerTorch(tq.QuantumModule):
    """Quantum‑enhanced transformer with optional quanvolution backbone."""
    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        num_blocks: int | None = None,
        ffn_dim: int | None = None,
        num_classes: int | None = None,
        dropout: float = 0.1,
        use_quanvolution: bool = False,
        n_wires_transformer: int = 8,
        n_wires_ffn: int = 8,
    ) -> None:
        super().__init__()
        if use_quanvolution:
            self.backbone = QuantumQuanvolutionClassifier(n_wires=n_wires_ffn, num_classes=num_classes or 10)
        else:
            assert vocab_size is not None and embed_dim is not None
            self.backbone = nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                PositionalEncoder(embed_dim),
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_wires_transformer,
                        n_wires_ffn,
                        dropout,
                    )
                    for _ in range(num_blocks)
                ],
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes if num_classes > 2 else 1),
            )
        self.use_quanvolution = use_quanvolution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

__all__ = [
    "QuantumFastBaseEstimator",
    "QuantumSelfAttention",
    "QuantumQuanvolutionFilter",
    "QuantumQuanvolutionClassifier",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerTorch",
]
