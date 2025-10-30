"""Quantum‑enhanced hybrid model that fuses a CNN, transformer, and quantum modules.

This implementation builds on the classical QFusionModel but replaces the optional
quantum bridge and transformer blocks with TorchQuantum circuits.  The quantum
components are lightweight 4‑qubit encoders that process feature vectors and
transformer token representations, enabling experiments that compare classical
and quantum expressivity side‑by‑side.
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
# 1. Classical CNN feature extractor – reused from the ML seed
# --------------------------------------------------------------------------- #
class _CNNFeatureExtractorQuantum(nn.Module):
    """Two‑layer CNN identical to the classical extractor."""

    def __init__(self, in_channels: int = 1, out_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self._flatten_size = 16 * 7 * 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.shape[0], -1)

# --------------------------------------------------------------------------- #
# 2. Quantum bridge – 4‑qubit encoder (QuantumNAT QML seed)
# --------------------------------------------------------------------------- #
class _QuantumBridgeQuantum(tq.QuantumModule):
    """Encodes a 4‑dim vector into a 4‑qubit state and returns a 4‑dim read‑out."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Simple encoder: a single RX gate per qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Variational parameters
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x: (batch, 4)
        self.encoder(q_device, x)
        for i, gate in enumerate(self.params):
            tq.RX(gate, wires=i)(q_device)  # apply parameterized RX
        return self.measure(q_device)

# --------------------------------------------------------------------------- #
# 3. Quantum attention block – lightweight 4‑qubit attention
# --------------------------------------------------------------------------- #
class _QuantumAttentionLayer(tq.QuantumModule):
    """Maps an embed_dim vector to a quantum state, applies a circuit, and returns a vector."""

    def __init__(self, embed_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = n_wires
        # Linear map to match qubit count
        self.input_linear = nn.Linear(embed_dim, n_wires, bias=False)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.output_linear = nn.Linear(n_wires, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x: (batch, embed_dim)
        q_input = self.input_linear(x)  # (batch, n_wires)
        self.encoder(q_device, q_input)
        for i, gate in enumerate(self.params):
            tq.RX(gate, wires=i)(q_device)
        out = self.measure(q_device)  # (batch, n_wires)
        return self.output_linear(out)

# --------------------------------------------------------------------------- #
# 4. Quantum feed‑forward block – 4‑qubit circuit
# --------------------------------------------------------------------------- #
class _QuantumFeedForwardLayer(tq.QuantumModule):
    """Quantum feed‑forward that maps embed_dim → embed_dim via a 4‑qubit circuit."""

    def __init__(self, embed_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = n_wires
        self.input_linear = nn.Linear(embed_dim, n_wires, bias=False)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.output_linear = nn.Linear(n_wires, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        q_input = self.input_linear(x)
        self.encoder(q_device, q_input)
        for i, gate in enumerate(self.params):
            tq.RY(gate, wires=i)(q_device)
        out = self.measure(q_device)
        return self.output_linear(out)

# --------------------------------------------------------------------------- #
# 5. Quantum transformer block – combines attention and feed‑forward
# --------------------------------------------------------------------------- #
class _QuantumTransformerBlock(tq.QuantumModule):
    """Transformer block where both attention and feed‑forward are quantum."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = _QuantumAttentionLayer(embed_dim, n_wires=n_wires)
        self.ffn = _QuantumFeedForwardLayer(embed_dim, n_wires=n_wires)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        # reshape for quantum operations
        batch, seq_len, embed_dim = x.size()
        x_flat = x.view(batch * seq_len, embed_dim)
        qdev = q_device or tq.QuantumDevice(n_wires=4, bsz=batch * seq_len, device=x.device)
        attn_out = self.attn(x_flat, qdev).view(batch, seq_len, embed_dim)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x.view(batch * seq_len, embed_dim), qdev).view(batch, seq_len, embed_dim)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 6. Positional encoder – same as classical
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

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

# --------------------------------------------------------------------------- #
# 7. Hybrid quantum model
# --------------------------------------------------------------------------- #
class QFusionModel(tq.QuantumModule):
    """Hybrid CNN‑Transformer‑Quantum classifier using TorchQuantum modules.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        shape of input image (C, H, W).
    dropout : float, optional
        dropout probability for transformer layers.
    n_blocks : int, optional
        number of transformer blocks.
    n_heads : int, optional
        number of attention heads.
    ffn_dim : int, optional
        dimension of feed‑forward network.
    num_classes : int, optional
        number of output classes.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (1, 28, 28),
        dropout: float = 0.1,
        n_blocks: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        # 1. CNN feature extractor
        self.cnn = _CNNFeatureExtractorQuantum(in_channels=input_shape[0])

        # 2. Transformer stack (quantum)
        embed_dim = self.cnn._flatten_size
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[ _QuantumTransformerBlock(embed_dim, n_heads, ffn_dim, n_wires=4) for _ in range(n_blocks) ]
        )
        self.dropout = nn.Dropout(dropout)

        # 3. Quantum bridge
        self.quantum_bridge = _QuantumBridgeQuantum(n_wires=4)

        # 4. Classifier
        final_dim = embed_dim + 4  # concatenate quantum bridge output
        self.classifier = nn.Linear(final_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extract features
        features = self.cnn(x)  # (B, D)

        # 2. Prepare sequence for transformer
        seq = features.unsqueeze(1)  # (B, 1, D)
        seq = self.pos_encoder(seq)

        # 3. Transformer
        trans_out = self.transformer(seq)  # (B, 1, D)
        trans_out = trans_out.squeeze(1)   # (B, D)

        # 4. Quantum bridge
        pool = F.avg_pool2d(x, 6).view(x.shape[0], -1)  # (B, 16)
        q_input = pool[:, :4]  # (B, 4)
        q_device = tq.QuantumDevice(n_wires=4, bsz=x.shape[0], device=x.device)
        q_out = self.quantum_bridge(q_input, q_device)  # (B, 4)

        # 5. Combine and classify
        combined = torch.cat([trans_out, q_out], dim=-1)  # (B, D+4)
        logits = self.classifier(self.dropout(combined))
        return logits

__all__ = ["QFusionModel"]
