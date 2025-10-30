"""
Hybrid transformer module with quantum‑kernel attention.

The quantum implementation replaces classical dot‑product attention with a
variational circuit that evaluates a quantum kernel for each query‑key pair.
Feed‑forward layers are also quantum‑enhanced.  The API mirrors the
classical version to enable side‑by‑side experiments.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Optional


# --------------------------------------------------------------------------- #
# Quantum‑enhanced attention base
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """
    Abstract base for multi‑head attention.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Quantum modules for linear projections
# --------------------------------------------------------------------------- #

class _QLayer(tq.QuantumModule):
    """
    Generic projection layer that encodes a classical vector into a
    quantum state and measures all qubits.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Simple RX‑based encoding
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                     for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=wire)
        # Entangle the wires in a ring
        for wire in range(self.n_wires - 1):
            tqf.cnot(q_device, wires=[wire, wire + 1])
        tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
        return self.measure(q_device)


# --------------------------------------------------------------------------- #
# Quantum kernel for similarity
# --------------------------------------------------------------------------- #

class QuantumKernel(tq.QuantumModule):
    """
    Simple quantum kernel that evaluates the overlap between two classical
    vectors by encoding them with opposite phases.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x)
        # Apply negative y by reversing the ansatz
        for info in reversed(self.ansatz.gates):
            func = tqf.func_name_dict[info["func"]]
            func(self.q_device, wires=info["wires"],
                 params=-y[:, info["input_idx"]]
                 if tq.op_name_dict[info["func"]].num_params else None)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
# Quantum‑enhanced attention
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Attention that uses a quantum kernel to weight key‑value pairs.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = embed_dim
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = _QLayer(self.n_wires)
        self.k_layer = _QLayer(self.n_wires)
        self.v_layer = _QLayer(self.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.kernel = QuantumKernel()

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, embed_dim = x.size()
        # Flatten batch and sequence for projection
        tokens = x.reshape(-1, embed_dim)

        q = self.q_layer(tokens, self.q_device).reshape(batch_size, seq_len, embed_dim)
        k = self.k_layer(tokens, self.q_device).reshape(batch_size, seq_len, embed_dim)
        v = self.v_layer(tokens, self.q_device).reshape(batch_size, seq_len, embed_dim)

        # Compute kernel similarity for each token pair
        sim = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    sim[b, i, j] = self.kernel(q[b, i, :], k[b, j, :])

        if mask is not None:
            mask_exp = mask.unsqueeze(1).unsqueeze(2)
            sim = sim.masked_fill(mask_exp == 0, -1e9)

        attn_weights = F.softmax(sim, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        return self.combine_heads(out)


# Quantum‑compatible alias for classical attention
MultiHeadAttentionClassical = MultiHeadAttentionQuantum


# --------------------------------------------------------------------------- #
# Feed‑forward layers
# --------------------------------------------------------------------------- #

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """
    Quantum‑enhanced feed‑forward layer using a variational circuit.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]}
                 for idx in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                         for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int,
                 n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# Classical feed‑forward alias
FeedForwardClassical = FeedForwardQuantum


# --------------------------------------------------------------------------- #
# Transformer blocks
# --------------------------------------------------------------------------- #

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """
    Quantum‑enhanced transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_qubits_transformer: int,
                 n_qubits_ffn: int, n_qlayers: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, q_device=q_device)

        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                          n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# Classical transformer block alias
TransformerBlockClassical = TransformerBlockQuantum


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
# Combined transformer classifier
# --------------------------------------------------------------------------- #

class HybridTransformerKernel(nn.Module):
    """
    Transformer‑based classifier that can instantiate quantum blocks
    by specifying the number of qubits for the transformer and
    feed‑forward sub‑modules.
    """
    def __init__(self,
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
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(
                n_wires=max(n_qubits_transformer, n_qubits_ffn)
            )
            blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    q_device=q_device,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads,
                                         ffn_dim, dropout)
                for _ in range(num_blocks)
            ]

        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
    "HybridTransformerKernel",
]
