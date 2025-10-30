"""
Hybrid model that fuses a residual CNN backbone, classical self‑attention,
quantum attention, and an optional quantum LSTM cell.  The module is
fully trainable with PyTorch optimizers and can be used as a drop‑in
replacement for the original `QuantumNAT` class.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum blocks from the separate QML module
from.QuantumHybridModel_qml import QuantumAttention, QuantumLSTMCell


class QuantumHybridModel(nn.Module):
    """
    Hybrid classical‑quantum model that merges a residual CNN backbone,
    a classical self‑attention head, a quantum attention circuit,
    and an optional quantum LSTM layer.  The final classifier operates
    on the concatenated embeddings.
    """

    def __init__(self, num_classes: int = 4, use_q_lstm: bool = True) -> None:
        super().__init__()
        # ---------- CNN backbone ----------
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res = _ResBlock(16, 32)

        # ---------- Classical attention ----------
        self.classical_attention = ClassicalSelfAttention(embed_dim=32)

        # ---------- Quantum modules ----------
        self.quantum_attention = QuantumAttention(n_qubits=4)
        self.use_q_lstm = use_q_lstm
        if use_q_lstm:
            self.quantum_lstm = QuantumLSTMCell(input_dim=36, hidden_dim=32, n_qubits=32)

        # ---------- Classifier ----------
        self.classifier = nn.Linear(68, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        feats = self.backbone(x)          # (B, 16, H/4, W/4)
        feats = self.res(feats)           # (B, 32, H/4, W/4)
        feats = feats.mean((2, 3))        # global avg pool -> (B, 32)

        # Classical self‑attention
        cls_attn = self.classical_attention(feats.cpu().numpy())
        cls_attn = torch.as_tensor(cls_attn, device=feats.device, dtype=feats.dtype)

        # Quantum self‑attention
        q_attn = self.quantum_attention(feats)   # (B, 4)

        # Concatenate embeddings
        combined = torch.cat([feats, q_attn], dim=1)  # (B, 36)

        # Optional quantum LSTM
        if self.use_q_lstm:
            lstm_out, _ = self.quantum_lstm(combined.unsqueeze(1))
            lstm_out = lstm_out.squeeze(1)  # (B, 32)
            combined = torch.cat([combined, lstm_out], dim=1)  # (B, 68)

        logits = self.classifier(combined)
        return logits


# --------------------------------------------------------------------------- #
#  Helper classes
# --------------------------------------------------------------------------- #
class _ResBlock(nn.Module):
    """Simple residual block with two 3×3 convs and a 1×1 projection."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch!= out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.proj is not None else x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)


class ClassicalSelfAttention:
    """Pure‑Python self‑attention used as a classical baseline."""
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        # Simple dot‑product attention with random projections
        query = feats @ np.random.randn(self.embed_dim, self.embed_dim)
        key   = feats @ np.random.randn(self.embed_dim, self.embed_dim)
        scores = np.exp(query @ key.T / math.sqrt(self.embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores @ feats


__all__ = ["QuantumHybridModel"]
