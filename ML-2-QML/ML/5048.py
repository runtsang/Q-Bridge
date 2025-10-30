import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Gen375(nn.Module):
    """
    Hybrid classical model combining convolution, self‑attention, sampling, and regression.
    This class acts as a drop‑in replacement for the legacy `Conv` filter but expands
    the functionality by adding a classical self‑attention block, a small sampler
    network, and a regression head.  All components are fully PyTorch compatible.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, embed_dim: int = 4):
        super().__init__()
        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

        # Self‑attention parameters
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(embed_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Regression head
        self.regress = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Tensor of shape (B, 1, kernel_size, kernel_size).

        Returns
        -------
        out
            Regression output of shape (B,).
        probs
            Sampler probabilities of shape (B, 2).
        """
        # Convolution + sigmoid threshold
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        feat = activations.view(x.size(0), -1)  # (B, embed_dim)

        # Self‑attention
        query = feat @ self.rotation
        key = feat @ self.entangle
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ feat  # (B, embed_dim)

        # Sampler
        probs = F.softmax(self.sampler(attn_out), dim=-1)  # (B, 2)

        # Regression head
        out = self.regress(attn_out).squeeze(-1)  # (B,)

        return out, probs

__all__ = ["Gen375"]
