"""Classical self‑attention module inspired by Quantum‑NAT and the original SelfAttention seed.

It builds on convolutional feature extraction and a classical attention
mechanism that uses trainable rotation and entanglement parameters.
The same public interface (`run`) is available for both the classical
and quantum variants, allowing downstream experiments to swap between
them without code changes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHybridML(nn.Module):
    """Hybrid self‑attention that first extracts features with a small CNN
    and then applies a classical attention block parameterised by
    rotation_params and entangle_params."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.norm = nn.BatchNorm1d(embed_dim)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            1‑D array of length 3*embed_dim used to build a rotation matrix.
        entangle_params : np.ndarray
            1‑D array of length embed_dim used to build a key matrix.
        inputs : np.ndarray
            Batch of grayscale images, shape (B,1,28,28).

        Returns
        -------
        np.ndarray
            Attention‑weighted embedding of shape (B, embed_dim).
        """
        # Convert to tensor and extract CNN features
        x = torch.as_tensor(inputs, dtype=torch.float32)
        feats = self.features(x).view(x.shape[0], -1)          # (B, 784)
        # Project to embedding space
        proj = self.fc(feats)                                 # (B, embed_dim)
        # Build query/key matrices from trainable params
        rot_mat = torch.as_tensor(rotation_params.reshape(
            self.embed_dim, -1), dtype=torch.float32)
        ent_mat = torch.as_tensor(entangle_params.reshape(
            self.embed_dim, -1), dtype=torch.float32)
        query = torch.matmul(proj, rot_mat)                   # (B, k)
        key   = torch.matmul(proj, ent_mat)                   # (B, k)
        # Compute attention scores
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        # Weighted sum over projection vectors
        out = scores @ proj                                  # (B, embed_dim)
        return self.norm(out).detach().cpu().numpy()

def SelfAttentionHybrid() -> SelfAttentionHybridML:
    """Return a fully initialised classical hybrid attention module."""
    return SelfAttentionHybridML()

__all__ = ["SelfAttentionHybrid"]
