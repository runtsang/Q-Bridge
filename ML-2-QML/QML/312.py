"""Quantum‑enhanced transformer layers implemented with TorchQuantum.

The module mirrors the classical API but replaces the attention and feed‑forward
sub‑modules with parameter‑driven quantum circuits.  A hybrid block is also
provided, allowing a learnable gate to interpolate between the classical and
quantum paths.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Attention layers
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that maps projections through a parameter‑driven quantum circuit."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.q_layer = self._QLayer(n_wires=self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self._apply_q(x)
        q = self._apply_q(x)
        v = self._apply_q(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)

    def _apply_q(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to each head projection."""
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, self.d_k)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

# --------------------------------------------------------------------------- #
# Feed‑forward layers
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a parameter‑driven quantum circuit."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits=n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
# Transformer blocks
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=n_qubits_ffn, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockHybrid(TransformerBlockBase):
    """Hybrid transformer block that mixes classical and quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        gate_init: float = 0.5,
        res_scale_init: float = 1.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn_classical = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn_classical = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        if use_quantum:
            self.attn_quantum = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
            self.ffn_quantum = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=8, dropout=dropout)
        else:
            self.attn_quantum = self.attn_classical
            self.ffn_quantum = self.ffn_classical
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate)
        # Attention mixing
        attn_cls = self.attn_classical(x)
        attn_q = self.attn_quantum(x)
        attn_out = gate * attn_q + (1 - gate) * attn_cls
        x = self.norm1(x + self.dropout(attn_out) * self.res_scale)

        # Feed‑forward mixing
        ffn_cls = self.ffn_classical(x)
        ffn_q = self.ffn_quantum(x)
        ffn_out = gate * ffn_q + (1 - gate) * ffn_cls
        return self.norm2(x + self.dropout(ffn_out) * self.res_scale)

# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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

# --------------------------------------------------------------------------- #
# Text classifier
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting hybrid blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_hybrid: bool = False,
        use_quantum: bool = False,
        gate_init: float = 0.5,
        res_scale_init: float = 1.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_hybrid:
            blocks = [
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout=dropout,
                    use_quantum=use_quantum,
                    gate_init=gate_init,
                    res_scale_init=res_scale_init,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_joint(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 1,
):
    """
    Simple training loop that performs joint optimisation of the model.
    """
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
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
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
    "train_joint",
]
