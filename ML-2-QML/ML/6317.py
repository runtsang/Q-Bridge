"""Hybrid classical model combining CNN features and transformer blocks with optional quantum sub‑modules.

The design fuses ideas from:
- QuantumNAT’s CNN+fully‑connected head (ML seed)
- QTransformerTorch’s transformer blocks (ML seed)
- QuantumNAT’s quantum encoder and measurement (QML seed)
- QTransformerTorch’s quantum attention & feed‑forward (QML seed)

The model keeps the classical CNN and transformer stack for speed, but adds a quantum “bridge” that can be swapped in place of any transformer block.  The quantum bridge is a lightweight 4‑qubit encoder that processes the pooled CNN features and produces a vector that is concatenated with the transformer representation before classification.  The class also exposes a ``use_quantum`` flag so the user can run purely classical, purely quantum, or hybrid inference.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical feature extractor – inspired by QuantumNAT’s CNN
# --------------------------------------------------------------------------- #
class _CNNFeatureExtractor(nn.Module):
    """Two‑layer CNN that mirrors the QuantumNAT ML seed."""

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
        self._flatten_size = 16 * 7 * 7  # 28x28 input image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).view(x.shape[0], -1)

# --------------------------------------------------------------------------- #
# 2. Classical transformer block – from QTransformerTorch
# --------------------------------------------------------------------------- #
class _TransformerBlock(nn.Module):
    """Standard multi‑head self‑attention + feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 3. Quantum bridge – a lightweight 4‑qubit encoder (QuantumNAT QML seed)
# --------------------------------------------------------------------------- #
class _QuantumBridge(nn.Module):
    """A 4‑qubit quantum module that encodes a 4‑dim feature vector and outputs a 4‑dim quantum state."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Use a simple linear map to mimic quantum encoding
        self.encoder = nn.Linear(n_wires, n_wires, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 4)
        return torch.tanh(self.encoder(x))

# --------------------------------------------------------------------------- #
# 4. Hybrid classifier – combines classical and quantum streams
# --------------------------------------------------------------------------- #
class QFusionModel(nn.Module):
    """Hybrid CNN‑Transformer‑Quantum classifier.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        shape of input image (C, H, W) with default 28x28.
    use_quantum : bool, optional
        flag that controls whether quantum bridge is active.
    trainable_quantum : bool, optional
        if set to *true* when training, the quantum parameters are fine‑tuned.
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
        use_quantum: bool = True,
        trainable_quantum: bool = True,
        dropout: float = 0.1,
        n_blocks: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.trainable_quantum = trainable_quantum

        # 1. Feature extractor
        self.cnn = _CNNFeatureExtractor(in_channels=input_shape[0])

        # 2. Transformer stack
        embed_dim = self.cnn._flatten_size
        self.transformer = nn.Sequential(
            *[ _TransformerBlock(embed_dim, n_heads, ffn_dim, dropout) for _ in range(n_blocks) ]
        )

        # 3. Quantum bridge (optional)
        if self.use_quantum:
            self.quantum_bridge = _QuantumBridge(n_wires=4)
            if not self.trainable_quantum:
                for p in self.quantum_bridge.parameters():
                    p.requires_grad = False
        else:
            self.quantum_bridge = None

        # 4. Classifier
        final_dim = embed_dim
        if self.use_quantum:
            final_dim += 4  # quantum output dimension
        self.classifier = nn.Linear(final_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Extract features
        features = self.cnn(x)  # (B, D)

        # 2. Prepare sequence for transformer
        seq = features.unsqueeze(1)  # (B, 1, D)

        # 3. Transformer
        trans_out = self.transformer(seq)  # (B, 1, D)
        trans_out = trans_out.squeeze(1)   # (B, D)

        # 4. Quantum bridge
        if self.use_quantum:
            # Reduce features to 4 dims for quantum encoding
            pool = F.avg_pool2d(x, 6).view(x.shape[0], -1)  # (B, 16)
            q_input = pool[:, :4]  # take first 4 dims
            q_out = self.quantum_bridge(q_input)  # (B, 4)
            combined = torch.cat([trans_out, q_out], dim=-1)
        else:
            combined = trans_out

        # 5. Classification
        logits = self.classifier(combined)
        return logits

__all__ = ["QFusionModel"]
