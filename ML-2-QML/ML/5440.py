from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["HybridSelfAttention"]

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention module.

    Architecture
    ------------
    * Convolutional encoder (mirrors the CNN backbone of QuantumNAT)
    * Linear projection to an embedding space
    * Optional attention weighting using externally supplied rotation/entangle
      parameters
    * Depth‑wise feed‑forward network (depth configurable)
    * Final linear classifier

    Parameters
    ----------
    embed_dim : int
        Dimension of the self‑attention embedding.
    conv_channels : int, default 8
        Number of channels in the first conv layer.
    depth : int, default 2
        Number of feed‑forward layers after attention.
    """
    def __init__(self, embed_dim: int, conv_channels: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Convolutional encoder (QuantumNAT style)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flattened feature size after two 2×2 poolings on 28×28 input
        feat_dim = conv_channels * 2 * 7 * 7
        self.proj = nn.Linear(feat_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

        # Self‑attention (single‑head for clarity)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

        # Feed‑forward stack
        self.feedforward = nn.Sequential(
            *[nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU()) for _ in range(depth)]
        )

        # Final classifier
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Forward pass producing logits.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, 1, 28, 28).
        rotation_params : np.ndarray, optional
            Rotation matrix for query projection, shape (embed_dim, embed_dim).
        entangle_params : np.ndarray, optional
            Entanglement matrix for key projection, shape (embed_dim, embed_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 2).
        """
        # Encoder
        feats = self.encoder(x)          # (B, C, H, W)
        flat = feats.view(feats.size(0), -1)  # (B, feat_dim)
        embed = self.proj(flat).unsqueeze(1)  # (B, 1, embed_dim)

        # Optional parameterised projections
        if rotation_params is not None:
            rot = torch.from_numpy(rotation_params).float()
            embed = torch.bmm(embed, rot)  # (B, 1, embed_dim)
        if entangle_params is not None:
            ent = torch.from_numpy(entangle_params).float()
            key = torch.bmm(embed, ent)  # (B, 1, embed_dim)
        else:
            key = embed

        # Self‑attention
        attn_out, _ = self.attention(embed, key, key)  # (B, 1, embed_dim)

        # Feed‑forward
        ff_out = self.feedforward(attn_out.squeeze(1))  # (B, embed_dim)

        # Classification
        logits = self.classifier(ff_out)  # (B, 2)
        return logits

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper to run the model on NumPy inputs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation matrix for query projection.
        entangle_params : np.ndarray
            Entanglement matrix for key projection.
        inputs : np.ndarray
            Batch of images, shape (B, 1, 28, 28).

        Returns
        -------
        np.ndarray
            Softmax probabilities, shape (B, 2).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                torch.from_numpy(inputs).float(),
                rotation_params=rotation_params,
                entangle_params=entangle_params,
            )
            probs = F.softmax(logits, dim=-1)
        return probs.numpy()
