from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """
    Classical 2‑D convolutional filter that emulates a quanvolution layer.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, out_channels: int = 4) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # shape (batch, out_channels)

class ClassicalSelfAttention(nn.Module):
    """
    Standard self‑attention block that uses rotation and entanglement parameters.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: torch.Tensor) -> torch.Tensor:
        rot = torch.tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32, device=inputs.device)
        ent = torch.tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32, device=inputs.device)
        query = inputs @ rot
        key   = inputs @ ent
        value = inputs
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class TransformerBlock(nn.Module):
    """
    Classical transformer block (attention + feed‑forward) used as a post‑processing layer.
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

class HybridSelfAttention(nn.Module):
    """
    Unified self‑attention module that chains a convolutional filter, a classical self‑attention block,
    and a transformer for richer representations. A lightweight binary classifier is appended.
    """
    def __init__(self,
                 embed_dim: int = 4,
                 conv_kernel: int = 2,
                 num_heads: int = 2,
                 ffn_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, out_channels=embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : array_like
            Parameters for the rotation gates used in the self‑attention block.
        entangle_params : array_like
            Parameters for the entanglement gates used in the self‑attention block.
        inputs : array_like
            Input tensor of shape (batch, features) or (H, W) for image‑like data.

        Returns
        -------
        np.ndarray
            Binary probability for each sample in the batch.
        """
        x = torch.tensor(inputs, dtype=torch.float32)
        # Convolutional preprocessing
        x = self.conv(x)
        # Self‑attention
        x = self.attention(rotation_params, entangle_params, x)
        # Transformer
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        # Classification
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()

def SelfAttention() -> HybridSelfAttention:
    """
    Factory function that mirrors the original API.
    """
    return HybridSelfAttention()
