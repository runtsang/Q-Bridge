"""Quantum‑enabled transformer classifier that mirrors the classical API."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


# ---------- Quantum Self‑Attention ----------
class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑parameterised multi‑head attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_layer = self._build_quantum_layer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)

    def _build_quantum_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [i], "func": "rx", "wires": [i]}
                        for i in range(n_wires)
                    ]
                )
                self.parameters = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                return self.measure(q_device)

        return QLayer(n_wires=8)  # fixed width for demonstration

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0))
                head_outputs.append(self.q_layer(qdev, head))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1).view(B, T, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k_q = self._apply_quantum(k)
        q_q = self._apply_quantum(q)
        v_q = self._apply_quantum(v)

        scores = torch.matmul(q_q, k_q.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_q)
        return self.combine_heads(out)


# ---------- Quantum Feed‑Forward ----------
class FeedForwardQuantum(nn.Module):
    """Feed‑forward network implemented with a small quantum circuit."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.q_layer = self._build_quantum_layer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def _build_quantum_layer(self, n_qubits: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [i], "func": "ry", "wires": [i]}
                        for i in range(n_wires)
                    ]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                return self.measure(q_device)
        return QLayer(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0))
            outputs.append(self.q_layer(qdev, token))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(torch.relu(out))


# ---------- Quantum Transformer Block ----------
class TransformerBlockQuantum(nn.Module):
    """Quantum‑enabled transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------- Positional Encoding ----------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding, identical to the classical version."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------- Quantum Kernel ----------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two vectors via a simple variational circuit."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Quantum RBF‑style kernel using the ansatz above."""

    def __init__(self) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q_device = self.ansatz.q_device.copy(bsz=x.size(0))
        self.ansatz(q_device, x, y)
        return torch.abs(q_device.states.view(-1)[0])


# ---------- Unified Classifier ----------
class UnifiedTransformerClassifier(nn.Module):
    """
    Quantum‑enabled transformer text classifier that shares the same public API as
    the classical counterpart defined in the ML module.  The class can optionally
    replace the linear head with a quantum kernel against a set of learnable
    class prototypes.
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
        use_kernel: bool = False,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.use_kernel = use_kernel
        if use_kernel:
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
            self.kernel = QuantumKernel()
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)  # [B, E]
        x = self.dropout(x)

        if self.use_kernel:
            sims = torch.stack(
                [self.kernel(x, proto) for proto in self.prototypes], dim=1
            )
            return sims
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumKernel",
    "UnifiedTransformerClassifier",
]
