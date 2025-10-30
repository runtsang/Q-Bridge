import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen180(nn.Module):
    """
    Classical self‑attention module that incorporates a convolutional
    front‑end, a self‑attention block, and a quantum‑inspired expectation
    head.  The class mirrors the quantum interface so that the same
    callable can be used in hybrid experiments.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shift: float = np.pi / 2,
        eps: float = 1e-3,
    ):
        super().__init__()
        # Convolutional filter (drop‑in replacement for quanvolution)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.threshold = threshold

        # Self‑attention parameters
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Quantum‑inspired head
        self.linear = nn.Linear(embed_dim, 1)
        self.shift = shift
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output probabilities of shape (batch, 1).
        """
        # 1. Convolutional feature extraction
        feat = self.conv(x)  # (B, D, H, W)
        B, D, H, W = feat.shape
        seq = feat.view(B, D, -1).transpose(1, 2)  # (B, seq_len, D)

        # 2. Classical self‑attention
        query = torch.matmul(seq, self.rotation_params)          # (B, seq, D)
        key   = torch.matmul(seq, self.entangle_params)          # (B, seq, D)
        scores = F.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(D),
            dim=-1,
        )  # (B, seq, seq)
        attn_out = torch.matmul(scores, seq)  # (B, seq, D)

        # 3. Global pooling
        pooled = attn_out.mean(dim=1)  # (B, D)

        # 4. Quantum‑inspired expectation head
        logits = self.linear(pooled)  # (B, 1)
        # Finite‑difference estimate of the gradient w.r.t. logits
        plus  = self.linear(pooled + self.eps)
        minus = self.linear(pooled - self.eps)
        grad  = (plus - minus) / (2 * self.eps)
        out = torch.sigmoid(logits + self.shift * grad)
        return out

__all__ = ["SelfAttentionGen180"]
