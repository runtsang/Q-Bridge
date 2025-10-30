"""Self‑attention module that fuses classical convolution, multi‑head attention and a fully‑connected post‑processor."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionGen412(nn.Module):
    """
    Classical self‑attention with optional parameter‑driven query/key/value construction,
    built on a lightweight convolutional feature extractor and a fully‑connected head.
    """

    def __init__(self,
                 embed_dim: int = 4,
                 n_heads: int = 2,
                 conv_features: int = 16,
                 fc_features: int = 32,
                 device: str | torch.device = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.device = device

        # Convolutional feature extractor (mirrors QCNNModel style)
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_features, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Linear layers to produce query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** -0.5

        # Fully‑connected post‑processing (mimics FCL)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_features),
            nn.Tanh(),
            nn.Linear(fc_features, embed_dim),
        )

        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: np.ndarray | None = None,
                entangle_params: np.ndarray | None = None) -> torch.Tensor:
        """
        Run a forward pass.

        Parameters
        ----------
        inputs: torch.Tensor
            Shape (batch, seq_len).  The module expects a 1‑D sequence per batch.
        rotation_params: np.ndarray | None
            Optional 2‑D array of shape (embed_dim, embed_dim) used to compute the
            query matrix.  When ``None`` a learnable linear layer is used.
        entangle_params: np.ndarray | None
            Optional 2‑D array of shape (embed_dim, embed_dim) used to compute the
            key matrix.  When ``None`` a learnable linear layer is used.
        """
        # Feature extraction
        x = self.conv(inputs.unsqueeze(1))  # (batch, embed_dim, seq_len')
        x = x.permute(0, 2, 1)             # (batch, seq_len', embed_dim)

        # Query / Key / Value
        if rotation_params is not None:
            # Use supplied parameters to build query
            rot = torch.as_tensor(rotation_params, dtype=torch.float32, device=self.device)
            Q = torch.matmul(x, rot.t())
        else:
            Q = self.query(x)

        if entangle_params is not None:
            key = torch.as_tensor(entangle_params, dtype=torch.float32, device=self.device)
            K = torch.matmul(x, key.t())
        else:
            K = self.key(x)

        V = self.value(x)

        # Scaled dot‑product attention
        scores = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = torch.matmul(scores, V)

        # Post‑processing
        out = self.fc(out)
        out = self.norm(out)

        return out

__all__ = ["SelfAttentionGen412"]
