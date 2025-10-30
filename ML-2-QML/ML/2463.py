import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention module operating on flattened feature maps."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ V

class QFCModel(nn.Module):
    """Hybrid classical CNN + self‑attention model."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)            # (bsz, 16, 7, 7)
        flattened = features.view(bsz, -1)     # (bsz, 784)
        seq_len = flattened.shape[1] // self.attention.embed_dim
        flattened_reshaped = flattened.view(bsz, seq_len, self.attention.embed_dim)
        attn_out = self.attention(flattened_reshaped)  # (bsz, seq_len, embed_dim)
        attn_flat = attn_out.view(bsz, -1)           # (bsz, 784)
        out = self.fc(attn_flat)
        return self.norm(out)

__all__ = ["QFCModel"]
