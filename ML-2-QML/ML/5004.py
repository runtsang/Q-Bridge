"""Hybrid self‑attention module that integrates classical attention,
fully‑connected layers, LSTM gating and a patch‑based convolution
filter inspired by the original SelfAttention, FCL, QLSTM and
Quanvolution seeds."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridSelfAttention(nn.Module):
    """
    A hybrid classical block that combines:

    * Self‑attention with query/key/value linear projections.
    * A simple LSTM cell that processes the attention output.
    * A fully‑connected head that collapses the hidden state to scalar.
    * An optional convolutional patch extractor (Quanvolution style)
      when the input is an image.

    Parameters
    ----------
    embed_dim : int
        Dimension of the feature embeddings.
    n_patches : int, optional
        If > 1, the module expects an image of shape
        (batch, 1, 28, 28) and applies a 2×2 patch extractor.
    """

    def __init__(self, embed_dim: int, n_patches: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches = n_patches

        # Attention projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # LSTM cell to process the attention sequence
        self.lstm_cell = nn.LSTMCell(embed_dim, embed_dim)

        # Final fully‑connected head
        self.out_head = nn.Linear(embed_dim, 1)

        # Optional convolutional patch extractor
        if n_patches > 1:
            self.patch_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            # 28×28 image -> 14×14 patches, each represented by 4 values
            patch_dim = 4 * (28 // 2) ** 2
            self.patch_fc = nn.Linear(patch_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. For sequence data: (batch, seq_len, embed_dim).
            For images: (batch, 1, 28, 28) when ``n_patches`` > 1.

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, 1) for sequences or
            (batch, 1) for images.
        """
        # Patch extraction for image inputs
        if self.n_patches > 1 and x.dim() == 4:
            # Apply 2×2 patch convolution
            patches = self.patch_conv(x)  # (batch, 4, 14, 14)
            patches = patches.view(x.size(0), -1)  # flatten
            x = self.patch_fc(patches)  # map to embed_dim

        # Self‑attention
        Q = self.query_proj(x)          # (batch, seq_len, embed_dim)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1
        )
        attn_out = torch.bmm(scores, V)  # (batch, seq_len, embed_dim)

        # LSTM gating over the sequence
        batch, seq_len, _ = attn_out.size()
        h = torch.zeros(batch, self.embed_dim, device=x.device)
        c = torch.zeros(batch, self.embed_dim, device=x.device)
        lstm_out = []
        for t in range(seq_len):
            h, c = self.lstm_cell(attn_out[:, t, :], (h, c))
            lstm_out.append(h.unsqueeze(1))
        lstm_out = torch.cat(lstm_out, dim=1)  # (batch, seq_len, embed_dim)

        # Fully‑connected head
        out = self.out_head(lstm_out)  # (batch, seq_len, 1)
        return out


__all__ = ["HybridSelfAttention"]
