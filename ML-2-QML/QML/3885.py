"""Quantum‑enhanced transformer for the hybrid model.

This module implements the same public class name, but the transformer
backbone uses quantum modules for attention and feed‑forward.  The
classical CNN encoder is copied from the ML version for self‑contained
operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Classical CNN feature extractor (identical to the ML version)
# --------------------------------------------------------------------------- #
class CNNFeatureExtractor(nn.Module):
    """Convolutional backbone used to extract image features."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int] = (8, 16),
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# --------------------------------------------------------------------------- #
#  Quantum attention block
# --------------------------------------------------------------------------- #
class QuantumAttentionBlock(nn.Module):
    """Multi‑head attention realised by small quantum circuits."""

    def __init__(self, embed_dim: int, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each token vector into qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        # One trainable RX per wire
        self.rx_gates = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, E]  (E == embed_dim)
        batch, seq_len, _ = x.shape
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=batch * seq_len, device=x.device
        )
        flattened = x.reshape(batch * seq_len, -1)
        self.encoder(qdev, flattened)
        self.random_layer(qdev)
        for wire, gate in enumerate(self.rx_gates):
            gate(qdev, wires=wire)
        outputs = self.measure(qdev)  # [B*T, n_wires]
        return outputs.reshape(batch, seq_len, self.n_wires)


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward block
# --------------------------------------------------------------------------- #
class QuantumFeedForwardBlock(nn.Module):
    """Feed‑forward network realised by a quantum circuit."""

    def __init__(self, embed_dim: int, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.ry_gates = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear layers to recover the original embedding size
        self.linear1 = nn.Linear(n_wires, 64)
        self.linear2 = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=batch * seq_len, device=x.device
        )
        flattened = x.reshape(batch * seq_len, -1)
        self.encoder(qdev, flattened)
        self.random_layer(qdev)
        for wire, gate in enumerate(self.ry_gates):
            gate(qdev, wires=wire)
        outputs = self.measure(qdev)  # [B*T, n_wires]
        outputs = outputs.reshape(batch, seq_len, self.n_wires)
        outputs = self.linear1(outputs)
        outputs = F.relu(outputs)
        outputs = self.linear2(outputs)
        return outputs


# --------------------------------------------------------------------------- #
#  Quantum transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    """Single transformer encoder block using quantum attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_wires: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # For simplicity we set n_wires == embed_dim; the attention and FFN
        # will output tensors of shape [B, T, embed_dim].
        self.attn = QuantumAttentionBlock(embed_dim, n_wires=n_wires)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = QuantumFeedForwardBlock(embed_dim, n_wires=n_wires)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class QuantumHybridNAT(nn.Module):
    """Hybrid CNN + transformer classifier with quantum sub‑modules."""

    def __init__(
        self,
        image_channels: int = 1,
        num_classes: int = 4,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = CNNFeatureExtractor(image_channels)
        self.embed = nn.Linear(16 * 7 * 7, embed_dim)
        self.transformer = nn.ModuleList(
            [
                QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, n_wires=embed_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)          # [B, 16, 7, 7]
        flat = features.view(features.size(0), -1)  # [B, 16*7*7]
        embedded = self.embed(flat).unsqueeze(1)    # [B, 1, embed_dim]
        for layer in self.transformer:
            embedded = layer(embedded)             # [B, 1, embed_dim]
        pooled = embedded.mean(dim=1)              # [B, embed_dim]
        out = self.classifier(pooled)
        return out


__all__ = ["QuantumHybridNAT"]
