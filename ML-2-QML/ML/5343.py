"""HybridSelfAttention: classical self‑attention module with optional LSTM gating, fully‑connected head, and quanvolution preprocessing."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention module that optionally:
      * applies an LSTM gate to the sequence
      * uses a fully‑connected linear head
      * preprocesses image data with a convolutional filter (quanvolution)
    The API mirrors the original SelfAttention.run signature.
    """

    def __init__(
        self,
        embed_dim: int,
        n_lstm_layers: int = 0,
        use_fc: bool = True,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_lstm = n_lstm_layers > 0
        self.use_fc = use_fc
        self.use_conv = use_conv

        # LSTM gating
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                num_layers=n_lstm_layers,
                batch_first=True,
            )

        # Fully‑connected head
        if self.use_fc:
            self.fc = nn.Linear(embed_dim, embed_dim)

        # Convolution for image preprocessing (quanvolution)
        if self.use_conv:
            # 1 channel input → 4 output channels, kernel 2x2 stride 2
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.linear_head = nn.Linear(4 * 14 * 14, embed_dim)

        # Attention projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. ``inputs`` can be:
          * (batch, seq_len, embed_dim) for sequence data
          * (batch, 1, 28, 28) for image data when ``use_conv`` is True
        """
        # Image path
        if self.use_conv and inputs.dim() == 4:
            x = self.conv(inputs)
            x = x.view(x.size(0), -1)
            if self.use_fc:
                x = self.linear_head(x)
            return x

        # Sequence path
        x = inputs
        if self.use_lstm:
            x, _ = self.lstm(x)

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1
        )
        attn_out = torch.matmul(scores, v)

        if self.use_fc:
            attn_out = self.fc(attn_out)

        return attn_out

    def run(
        self,
        rotation_params: Optional[torch.Tensor] = None,
        entangle_params: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compatibility wrapper matching the original SelfAttention.run signature.
        Parameters ``rotation_params`` and ``entangle_params`` are ignored for the classical implementation.
        """
        if inputs is None:
            raise ValueError("``inputs`` must be provided.")
        out = self.forward(inputs)
        return out.detach().numpy()


def SelfAttention(
    embed_dim: int = 4,
    n_lstm_layers: int = 0,
    use_fc: bool = True,
    use_conv: bool = False,
) -> HybridSelfAttention:
    """Factory returning a hybrid classical self‑attention instance."""
    return HybridSelfAttention(embed_dim, n_lstm_layers, use_fc, use_conv)


__all__ = ["HybridSelfAttention", "SelfAttention"]
