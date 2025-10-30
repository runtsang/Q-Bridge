"""Quantum‑enhanced regression model using TorchQuantum."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import math
from typing import Optional, List, Tuple

# --------------------------------------------------------------------------- #
# 1. Dataset & data generation (quantum variant)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    together with a target y = sin(2θ) cos(φ).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    theta = 2 * np.pi * np.random.rand(samples)
    phi = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(theta[i]) * omega_0 + np.exp(1j * phi[i]) * np.sin(theta[i]) * omega_1
    labels = np.sin(2 * theta) * np.cos(phi)
    return states, labels

class QuantumDataset(torch.utils.data.Dataset):
    """Dataset that provides quantum states as inputs."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# 2. Quantum patch‑wise feature extractor
# --------------------------------------------------------------------------- #

class QuantumPatchFilter(tq.QuantumModule):
    """
    Random two‑qubit quantum kernel applied to patches of size 4.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> torch.Tensor:
        """
        Encode a single patch and return the Pauli‑Z measurement.
        """
        self.encoder(qdev, patch)
        self.random_layer(qdev)
        return self.measure(qdev)

class QuantumFeatureExtractor(tq.QuantumModule):
    """
    Splits the classical feature vector into patches of size 4 and encodes each
    patch with a quantum circuit.  The concatenated measurements form a
    representation suitable for downstream classical layers.
    """
    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_filter = QuantumPatchFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, F) where F is divisible by patch_size.

        Returns
        -------
        features : torch.Tensor
            Shape (B, seq_len, n_wires) where seq_len = F / patch_size.
        """
        B, F = x.shape
        if F % self.patch_size!= 0:
            raise ValueError("Number of features must be divisible by patch_size")
        seq_len = F // self.patch_size
        patches = x.view(B, seq_len, self.patch_size)
        all_features: List[torch.Tensor] = []
        for i in range(seq_len):
            patch = patches[:, i, :]  # (B, patch_size)
            qdev = tq.QuantumDevice(n_wires=self.patch_filter.n_wires, bsz=B, device=x.device)
            feat = self.patch_filter(qdev, patch)
            all_features.append(feat)
        return torch.stack(all_features, dim=1)  # (B, seq_len, n_wires)

# --------------------------------------------------------------------------- #
# 3. Transformer components (classical implementation reused)
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """
    Base class for multi‑head attention layers.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention implemented with PyTorch linear projections.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding dimension does not match layer size")
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        d_k = embed_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.combine_heads(out)

class FeedForwardBase(nn.Module):
    """
    Base class for feed‑forward networks.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """
    Two‑layer feed‑forward network with ReLU activation.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    """
    Base class for a transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """
    Classical transformer block consisting of multi‑head attention and feed‑forward.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
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

class TransformerBackbone(nn.Module):
    """
    Sequence of transformer blocks.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# --------------------------------------------------------------------------- #
# 4. Hybrid quantum‑enhanced regression model
# --------------------------------------------------------------------------- #

class QuantumRegression(nn.Module):
    """
    Quantum‑enhanced regression model that merges quantum feature extraction
    with a classical transformer backbone and a linear head.

    The class name mirrors the classical counterpart defined in the ML module.
    """
    def __init__(
        self,
        num_features: int,
        patch_size: int = 4,
        *,
        embed_dim: int | None = None,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_extractor = QuantumFeatureExtractor(patch_size)
        seq_len = num_features // patch_size
        embed_dim = embed_dim or self.feature_extractor.patch_filter.n_wires
        self.transformer = TransformerBackbone(embed_dim, num_heads, ffn_dim, num_blocks, dropout)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, num_features) with num_features divisible by patch_size.

        Returns
        -------
        output : torch.Tensor
            Shape (B,) regression outputs.
        """
        features = self.feature_extractor(x)          # (B, seq_len, embed_dim)
        features = self.pos_encoder(features)
        features = self.transformer(features)
        pooled = features.mean(dim=1)                 # (B, embed_dim)
        return self.head(pooled).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "QuantumDataset",
    "QuantumPatchFilter",
    "QuantumFeatureExtractor",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TransformerBackbone",
    "QuantumRegression",
]
