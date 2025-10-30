"""Quantum‑enhanced transformer with configurable quantum depth and noise."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class DropoutLayerNorm(nn.Module):
    """LayerNorm with dropout applied before normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dropout(x))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with a learnable bias."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)] + self.bias


class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention with a variational quantum circuit applied to the output."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qlayers: int = 1,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_qlayers = n_qlayers
        self.noise_scale = noise_scale

        # Classical attention backbone
        self.classical_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Quantum submodule
        self.q_device = tq.QuantumDevice(n_wires=embed_dim, device="cpu")
        self.q_layer = self._build_quantum_layer()

    def _build_quantum_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int, n_layers: int):
                super().__init__()
                self.n_wires = n_wires
                self.n_layers = n_layers
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                for _ in range(self.n_layers - 1):
                    for wire in range(self.n_wires - 1):
                        tqf.cnot(q_device, wires=[wire, wire + 1])
                    tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
                return self.measure(q_device)
        return QLayer(self.q_device.n_wires, self.n_qlayers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Classical multi‑head attention
        attn_out, _ = self.classical_attn(x, x, x, key_padding_mask=mask)
        # Quantum post‑processing on each token
        B, T, D = attn_out.shape
        outputs = []
        for token in attn_out.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            if self.noise_scale > 0.0:
                out += torch.randn_like(out) * self.noise_scale
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised via a variational quantum circuit."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
        n_qlayers: int = 1,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.noise_scale = noise_scale
        self.dropout = nn.Dropout(dropout)

        # Classical mapping to quantum input
        self.linear1 = nn.Linear(embed_dim, n_qubits)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        # Quantum submodule
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, device="cpu")
        self.q_layer = self._build_quantum_layer()

    def _build_quantum_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int, n_layers: int):
                super().__init__()
                self.n_wires = n_wires
                self.n_layers = n_layers
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                for _ in range(self.n_layers - 1):
                    for wire in range(self.n_wires - 1):
                        tqf.cnot(q_device, wires=[wire, wire + 1])
                    tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
                return self.measure(q_device)
        return QLayer(self.n_qubits, self.n_qlayers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear mapping to quantum input
        q_input = self.linear1(x)
        outputs = []
        for token in q_input.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            if self.noise_scale > 0.0:
                out += torch.randn_like(out) * self.noise_scale
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear2(self.dropout(out))
        return F.relu(out)


class TransformerBlockQuantum(nn.Module):
    """Transformer block that optionally uses quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qlayers_attn: int,
        n_qlayers_ffn: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = DropoutLayerNorm(embed_dim, dropout=dropout)
        self.norm2 = DropoutLayerNorm(embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        if n_qlayers_attn > 0:
            self.attn = MultiHeadAttentionQuantum(
                embed_dim, num_heads, dropout=dropout,
                n_qlayers=n_qlayers_attn, noise_scale=noise_scale
            )
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        if n_qlayers_ffn > 0:
            self.ffn = FeedForwardQuantum(
                embed_dim, ffn_dim, n_qubits_ffn, dropout=dropout,
                n_qlayers=n_qlayers_ffn, noise_scale=noise_scale
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention
        if isinstance(self.attn, MultiHeadAttentionQuantum):
            attn_out = self.attn(x, mask)
        else:
            attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed‑forward
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer with configurable quantum depth and noise."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        n_qlayers_attn: int = 0,
        n_qlayers_ffn: int = 0,
        n_qubits_ffn: int = 0,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qlayers_attn,
                    n_qlayers_ffn,
                    n_qubits_ffn,
                    dropout=dropout,
                    noise_scale=noise_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = ["HybridTransformer"]
