"""Hybrid transformer‑based text classifier with quantum sub‑modules.

This module implements the same public API as the classical version
but replaces the attention and feed‑forward layers with quantum
variational circuits.  The `TextClassifierHybrid` class can be used
directly with a `torchquantum.QuantumDevice` or any compatible
backend.  The implementation follows the same structure as the
original seed but adds a rotary positional encoder and a
learnable output projection.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ---------- Attention primitives ----------
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange tensor to (batch, heads, seq_len, d_k)."""
        batch_size = x.size(0)
        return (
            x.view(batch_size, -1, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention implemented with quantum circuits."""

    class _QLayer(tq.QuantumModule):
        """Quantum module that encodes a single head."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            # Entangle adjacent qubits to increase expressivity
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.n_wires_per_head = embed_dim // num_heads
        self.q_layer = self._QLayer(self.n_wires_per_head)
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def _apply_quantum_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum circuit to each head of the input."""
        batch, seq_len, embed_dim = x.size()
        # Split heads
        x_split = x.view(batch, seq_len, self.num_heads, self.n_wires_per_head)
        outputs = []
        for head in range(self.num_heads):
            head_tensor = x_split[:, :, head, :]  # (batch, seq_len, n_wires)
            # Flatten batch and sequence for quantum device
            flat = head_tensor.contiguous().view(-1, self.n_wires_per_head)
            qdev = self.q_device or tq.QuantumDevice(
                n_wires=self.n_wires_per_head,
                bsz=flat.size(0),
                device=flat.device,
            )
            out = self.q_layer(flat, qdev)
            out = out.view(batch, seq_len, self.n_wires_per_head)
            outputs.append(out)
        # Concatenate heads back
        return torch.stack(outputs, dim=2).view(batch, seq_len, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Classical linear projections to prepare quantum inputs
        k = self._apply_quantum_projection(x)
        q = self._apply_quantum_projection(x)
        v = self._apply_quantum_projection(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            x.size(0), -1, self.embed_dim
        )
        return self.combine_heads(attn_output)


# ---------- Feed‑forward primitives ----------
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward implemented with a quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# ---------- Transformer block ----------
class TransformerBlockBase(nn.Module):
    """Base transformer block."""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attention: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, q_device=tq.QuantumDevice(n_wires=n_qubits_attention)
        )
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------- Rotary positional encoding ----------
class RotaryPositionalEncoding(nn.Module):
    """Sinusoidal rotary positional encoding with learnable frequency."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.freq = nn.Parameter(torch.ones(embed_dim // 2) * 10000.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        inv_freq = 1.0 / (self.freq ** (torch.arange(0, self.embed_dim, 2, device=x.device) / self.embed_dim))
        sinusoid_inp = torch.einsum("i, j -> i j", position, inv_freq)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        x_even = x[:, :, ::2]
        x_odd = x[:, :, 1::2]
        x_rotated = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        return x_rotated


# ---------- Hybrid classifier ----------
class TextClassifierHybrid(nn.Module):
    """Hybrid transformer‑based text classifier with quantum sub‑modules.

    The default implementation uses quantum attention and feed‑forward
    layers.  The `use_quantum` flag is accepted for API compatibility
    but is ignored – the class always operates quantum‑wise.
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
        n_qubits_attention: int = 8,
        n_qubits_ffn: int = 8,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = RotaryPositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attention,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(self.output_proj(x))
        return self.classifier(x)

    def evaluate(self, dataloader, loss_fn, device: torch.device = torch.device("cpu")) -> dict:
        """Simple evaluation loop returning loss and accuracy."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                logits = self.forward(inputs)
                loss = loss_fn(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1) if logits.dim() > 1 else logits.squeeze()
                correct += (preds == targets).sum().item()
                total += inputs.size(0)
        return {"loss": total_loss / total, "accuracy": correct / total}


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "RotaryPositionalEncoding",
    "TextClassifierHybrid",
]
