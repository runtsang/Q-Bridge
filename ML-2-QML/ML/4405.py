"""QuantumHybridUnifiedModel – classical backbone with optional quantum kernel.

This module implements a unified model that can optionally use a quantum
kernel for patch extraction and can switch between a classical LSTM or a
classical Transformer encoder.  The API is compatible with the original
seed modules so that existing training scripts can be reused.

The design follows the *combination* scaling paradigm: each sub‑module
is independently scalable and can be swapped out at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPatchEncoder(nn.Module):
    """Convolutional patch extractor.

    When ``use_quantum_kernel`` is False (default) the module performs a
    standard 2×2 stride‑2 convolution that reduces the spatial
    resolution by a factor of two and outputs 4 feature maps.  The
    output is flattened into a 1‑D tensor of shape
    ``(batch, 4 * H/2 * W/2)``.
    """
    def __init__(self, in_channels: int = 1, use_quantum_kernel: bool = False) -> None:
        super().__init__()
        self.use_quantum_kernel = use_quantum_kernel
        if not self.use_quantum_kernel:
            self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        else:
            raise NotImplementedError("Quantum kernel path is only available in the quantum module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quantum_kernel:
            return self.conv(x).view(x.size(0), -1)
        raise RuntimeError("Quantum kernel path should not be used in the classical module.")


class HybridEncoder(nn.Module):
    """Sequence encoder that can be a classical LSTM or a classical Transformer."""
    def __init__(self, embed_dim: int, mode: str = "lstm", *, n_layers: int = 1) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.embed_dim = embed_dim
        if self.mode == "lstm":
            self.encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True, num_layers=n_layers)
        elif self.mode == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                       nhead=8,
                                                       dim_feedforward=4 * embed_dim,
                                                       dropout=0.1,
                                                       activation="relu")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            raise ValueError(f"Unsupported encoder mode {self.mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        if isinstance(self.encoder, nn.LSTM):
            out, _ = self.encoder(x)
            return out[:, -1, :]  # last hidden state
        else:
            return self.encoder(x).mean(dim=1)  # mean over sequence


class HybridClassifier(nn.Module):
    """Linear classification head."""
    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantumHybridUnifiedModel(nn.Module):
    """Unified model that stitches together the encoder, head and optional
    quantum sub‑modules.  The public API mirrors the original `QuantumNAT`
    and `Quanvolution` classes so that training scripts can call
    `model = QuantumHybridUnifiedModel(..., use_quantum_kernel=True)` etc.
    """
    def __init__(
        self,
        *,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_classes: int = 10,
        encoder_mode: str = "lstm",
        use_quantum_kernel: bool = False,
        n_qubits: int = 0,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = ConvPatchEncoder(in_channels, use_quantum_kernel=use_quantum_kernel)
        self.h_encoder = HybridEncoder(embed_dim, mode=encoder_mode, n_layers=n_layers)
        self.classifier = HybridClassifier(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolution / patch extraction
        patch_emb = self.encoder(x)  # shape: (batch, 4 * H/2 * W/2)
        # 2. Reshape to sequence for the encoder
        seq_len = patch_emb.shape[1] // self.h_encoder.embed_dim
        if seq_len * self.h_encoder.embed_dim!= patch_emb.shape[1]:
            raise ValueError(f"Patch embedding size {patch_emb.shape[1]} is not divisible by embed_dim {self.h_encoder.embed_dim}")
        patch_seq = patch_emb.view(x.size(0), seq_len, self.h_encoder.embed_dim)
        # 3. Encode with hybrid LSTM/Transformer
        encoded = self.h_encoder(patch_seq)
        # 4. Classifier head
        return self.classifier(encoded)
