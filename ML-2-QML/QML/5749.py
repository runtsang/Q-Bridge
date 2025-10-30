"""
Hybrid transformer that replaces classical sub‑modules with TorchQuantum variational circuits.
The architecture mirrors the classical version but uses quantum attention and feed‑forward
layers.  An optional graph‑based mask, derived from pairwise token similarity, can be
applied to the attention weights.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
import itertools

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Graph utilities (torch based)                                            #
# --------------------------------------------------------------------------- #
def _state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def _fidelity_adjacency(
    states: Iterable[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Quantum transformer components                                           #
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced multi‑head attention."""

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(self.n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: Tensor, q_device: tq.QuantumDevice) -> Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding does not match layer size")
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out, _ = self.attention(q, k, v, mask)
        return self.combine_heads(out)

    def _apply_quantum_heads(self, x: Tensor) -> Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device
                )
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)


class GraphAttentionQuantum(MultiHeadAttentionQuantum):
    """Quantum attention with a graph‑based mask."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, q_device=q_device)
        self.graph_threshold = graph_threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def _build_mask(self, x: Tensor) -> Tensor:
        norm_x = x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        sim = torch.matmul(norm_x, norm_x.transpose(-2, -1))
        mask = sim >= self.graph_threshold
        if self.secondary is not None:
            mask |= (sim >= self.secondary) & (sim < self.graph_threshold)
        return mask

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = self._build_mask(x)
        return super().forward(x, mask)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: Tensor, q_device: tq.QuantumDevice) -> Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockHybridQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = GraphAttentionQuantum(
            embed_dim,
            num_heads,
            dropout,
            graph_threshold,
            secondary,
            secondary_weight,
            q_device=q_device,
        )
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTransformerQuantum(nn.Module):
    """Text classifier that uses quantum sub‑modules and an optional graph mask."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [
                TransformerBlockHybridQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    q_device=q_device,
                    dropout=dropout,
                    graph_threshold=graph_threshold,
                    secondary=secondary,
                    secondary_weight=secondary_weight,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    graph_threshold,
                    secondary,
                    secondary_weight,
                )
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# Public alias for backward compatibility
HybridTransformer = HybridTransformerQuantum

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "GraphAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockHybridQuantum",
    "PositionalEncoder",
    "HybridTransformerQuantum",
    "HybridTransformer",
]
