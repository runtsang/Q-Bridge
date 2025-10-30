"""
Hybrid regression model for classical training.

The module exposes:
  * `RegressionDataset` – generates 2‑D superposition images with scalar targets.
  * `HybridRegressionModel` – a CNN → embedding → transformer → linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_images(num_samples: int, kernel_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 2‑D images from a superposition state.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    kernel_size : int
        Size of the square image (e.g. 8 for 8×8).

    Returns
    -------
    images : np.ndarray
        Shape (num_samples, kernel_size, kernel_size).
    labels : np.ndarray
        Shape (num_samples,).
    """
    # Build a simple superposition: |psi> = cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>
    n_wires = kernel_size ** 2
    thetas = 2 * np.pi * np.random.rand(num_samples)
    phis = 2 * np.pi * np.random.rand(num_samples)

    states = np.zeros((num_samples, 2 ** n_wires), dtype=complex)
    for i in range(num_samples):
        states[i] = np.cos(thetas[i]) * np.eye(1, 2 ** n_wires, 0)[0] + \
                    np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(1, 2 ** n_wires, -1)[0]
    # Label: a simple non‑linear function of the angles
    labels = np.sin(2 * thetas) * np.cos(phis)

    # Reshape to 2‑D images
    images = states.real.reshape(num_samples, kernel_size, kernel_size)
    return images, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset yielding 2‑D images and scalar targets.
    """
    def __init__(self, samples: int, kernel_size: int):
        self.images, self.labels = generate_superposition_images(samples, kernel_size)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {
            "image": torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0),  # (1, H, W)
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Positional encoding (copied from QTransformerTorch)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Transformer block (classical)
# --------------------------------------------------------------------------- #
class TransformerBlockClassical(nn.Module):
    """
    Standard transformer block with multi‑head attention and feed‑forward.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """
    CNN → linear embedding → transformer → linear head.
    """
    def __init__(
        self,
        kernel_size: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        # Simple CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Linear embedding from channel dimension to embed_dim
        self.embed = nn.Linear(16, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        # Transformer stack
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
                for _ in range(num_blocks)
            ]
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Shape (B, 1, H, W) – grayscale image.

        Returns
        -------
        torch.Tensor
            Predicted scalar, shape (B,).
        """
        # CNN feature extraction
        feat = self.cnn(image)          # (B, 16, H', W')
        B, C, H, W = feat.shape
        # Prepare sequence: flatten spatial dims, keep channel as feature dim
        seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, seq_len, C)
        # Embed to transformer dimension
        seq = self.embed(seq)           # (B, seq_len, embed_dim)
        seq = self.pos_encoder(seq)
        # Transformer
        seq = self.transformer(seq)
        # Pool and head
        pooled = seq.mean(dim=1)        # (B, embed_dim)
        out = self.fc(pooled).squeeze(-1)  # (B,)
        return out

__all__ = ["RegressionDataset", "HybridRegressionModel"]
