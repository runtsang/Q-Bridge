"""
QTransformerExtendedQuantum: quantum‑augmented transformer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """
    Base class for quantum attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_mask: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        if use_mask:
            self.mask = nn.Parameter(torch.ones(1, 1, embed_dim, embed_dim))

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return (
            x.view(batch, seq, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Quantum multi‑head attention using a parameter‑efficient ansatz.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Simple ansatz: rotate each qubit, entangle with CNOT chain
            self.encoders = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoders(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for i in range(self.n_qubits - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_mask: bool = False,
                 n_qubits: int = 8, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout, use_mask)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum layer to each token head.
        """
        batch, seq, _ = x.shape
        heads = x.view(batch, seq, self.num_heads, self.d_k)
        outputs = []
        for i in range(seq):
            token = heads[:, i, :, :].view(batch, self.num_heads, self.d_k)
            token_out = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                token_out.append(self.q_layer(head, qdev))
            outputs.append(torch.stack(token_out, dim=1))
        return torch.stack(outputs, dim=1).view(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Linear projections
        q = self._apply_quantum(x)
        k = self._apply_quantum(x)
        v = self._apply_quantum(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.use_mask:
            scores = scores * self.mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    """
    Base class for quantum feed‑forward.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """
    Quantum feed‑forward using a small ansatz.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoders = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoders(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for i in range(seq):
            token = x[:, i, :]
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """
    Quantum transformer block.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits: int,
        n_qlayers: int = 1,
        dropout: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, use_mask, n_qubits)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
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


class QTransformerExtendedQuantum(nn.Module):
    """
    Quantum‑augmented transformer‑based classifier.
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
        n_qubits: int = 8,
        n_qlayers: int = 1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits, n_qlayers, dropout, use_mask)
              for _ in range(num_blocks)]
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def set_params(
        self,
        *,
        dropout: Optional[float] = None,
        use_mask: Optional[bool] = None,
    ) -> None:
        """
        Update hyperparameters that affect the quantum blocks.
        """
        if dropout is not None:
            self.dropout.p = dropout
            for block in self.transformer:
                if hasattr(block, "attn"):
                    block.attn.dropout.p = dropout
                if hasattr(block, "ffn"):
                    block.ffn.dropout.p = dropout
        if use_mask is not None:
            for block in self.transformer:
                if hasattr(block, "attn"):
                    block.attn.use_mask = use_mask

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerExtendedQuantum",
]
