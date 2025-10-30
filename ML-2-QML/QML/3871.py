"""
Quantum‑enhanced transformer using TorchQuantum and a hybrid quantum head.
The implementation mirrors the classical version but replaces all
attention/FFN sub‑modules with their quantum counterparts when
requested.  The final classification head is a parameterised quantum
circuit that outputs the expectation of Pauli‑Z, enabling a fully
differentiable quantum‑classical interface.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TorchQuantum imports – they are optional; if unavailable the module will
# raise ImportError at import time, making the dependency explicit.
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:  # pragma: no cover
    raise ImportError("torchquantum must be installed to use QTransformerHybrid quantum mode")

# ------------------------------------------------------------------
# Quantum primitives
# ------------------------------------------------------------------
class QAttentionLayer(tq.QuantumModule):
    """Quantum attention head – each head is a small circuit on n_wires qubits."""
    def __init__(self, n_heads: int, n_wires: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_wires = n_wires
        # Encoder maps the classical embedding of a token to qubit rotations
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.parameters = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        # x : (batch, n_wires)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.parameters):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class QFeedForwardLayer(tq.QuantumModule):
    """Quantum feed‑forward block – a small quantum circuit followed by classical linear layers."""
    def __init__(self, n_qubits: int, ffn_dim: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, n_qubits)

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.parameters):
            gate(qdev, wires=wire)
        out = self.measure(qdev)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# ------------------------------------------------------------------
# Quantum‑aware transformer blocks
# ------------------------------------------------------------------
class QuantumAttention(nn.Module):
    """Handles the classical or quantum attention sub‑module."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, n_wires: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.q_layer = QAttentionLayer(num_heads, n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v)

    def _quantum_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum encoder to each token."""
        batch, seq, _ = x.size()
        proj = []
        for token in x.unbind(dim=1):  # seq
            token = token.view(token.size(0), self.num_heads, -1)
            head_outs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outs.append(self.q_layer(head, qdev))
            proj.append(torch.stack(head_outs, dim=1))
        return torch.stack(proj, dim=1)  # (batch, seq, num_heads, wires)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # quantum projections
        k_q = self._quantum_projection(k)
        q_q = self._quantum_projection(q)
        v_q = self._quantum_projection(v)
        out = self.attention(q_q, k_q, v_q, mask)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.combine_heads(out)


class QuantumFeedForward(nn.Module):
    """Quantum feed‑forward implemented with a small circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn = QFeedForwardLayer(n_qubits, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.ffn.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.ffn(token, qdev))
        out = torch.stack(outputs, dim=1)
        return out


class QuantumTransformerBlock(nn.Module):
    """Transformer block that can mix classical & quantum sub‑modules."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_attn: int = 0,
                 n_qubits_ffn: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if n_wires_attn > 0:
            self.attn = QuantumAttention(embed_dim, num_heads, dropout, n_wires_attn)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        if n_qubits_ffn > 0:
            self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed‑forward
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ------------------------------------------------------------------
# Hybrid quantum head
# ------------------------------------------------------------------
class QuantumHybridHead(nn.Module):
    """Parameterised quantum circuit that outputs the expectation of Pauli‑Z."""
    class _QuantumCircuit(tq.QuantumModule):
        def __init__(self, n_qubits: int = 1) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]}]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, in_features: int, n_qubits: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.circuit = self._QuantumCircuit(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to a single parameter per batch element
        param = self.linear(x).view(-1, 1)
        out = []
        for p in param:
            qdev = self.q_device.copy(bsz=1, device=p.device)
            out.append(self.circuit(p, qdev))
        return torch.stack(out, dim=0).squeeze(-1)


# ------------------------------------------------------------------
# Main quantum transformer
# ------------------------------------------------------------------
class QTransformerHybrid(nn.Module):
    """
    Full transformer with optional quantum sub‑modules.

    The class is API‑compatible with the classical version but every
    block can be switched to its quantum counterpart via the ``n_wires_attn`` and
    ``n_qubits_ffn`` arguments.  The classification head is a quantum
    expectation circuit that can be replaced by a classical sigmoid if desired.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 n_wires_attn: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qubits_head: int = 0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.ModuleList([PositionalEncoder(embed_dim)])
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                     n_wires_attn=n_wires_attn,
                                     n_qubits_ffn=n_qubits_ffn,
                                     dropout=dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if n_qubits_head > 0:
            self.head = QuantumHybridHead(embed_dim, n_qubits_head)
            self.classifier = nn.Identity()
        else:
            self.head = nn.Identity()
            self.classifier = nn.Linear(embed_dim, num_classes) if num_classes > 2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder[0](x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        if isinstance(self.classifier, nn.Identity):
            out = self.head(x)
            return torch.cat((out.unsqueeze(-1), 1 - out.unsqueeze(-1)), dim=-1)
        else:
            logits = self.classifier(x)
            return logits


__all__ = [
    "QuantumAttention",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "QuantumHybridHead",
    "QTransformerHybrid",
]
