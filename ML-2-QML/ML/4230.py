import torch
import torch.nn as nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """
    Lightweight dot‑product self‑attention module.
    """
    def __init__(self, embed_dim: int, input_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(input_dim, embed_dim)
        self.key   = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class QuantumNATGen221(nn.Module):
    """
    Classical counterpart of the hybrid Quantum‑NAT model.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Determine flattened feature dimension (28x28 input -> 7x7x16)
        self.flat_dim = 16 * 7 * 7
        # Self‑attention module
        self.attention = ClassicalSelfAttention(embed_dim=4, input_dim=self.flat_dim)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        attn_out = self.attention(flat)
        out = self.fc(attn_out)
        return self.norm(out)

__all__ = ["QuantumNATGen221"]
