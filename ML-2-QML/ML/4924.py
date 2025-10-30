import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """Classical self‑attention block with trainable rotation and entanglement matrices."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, self.rotation_params)
        key   = torch.matmul(inputs, self.entangle_params)
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

class KernalAnsatz(nn.Module):
    """RBF kernel ansatz used in the classical kernel module."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a single forward call for a pair of vectors."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class HybridNATModel(nn.Module):
    """
    Hybrid classical model that concatenates:
    * Convolutional feature extractor
    * Learnable embedding projection
    * Self‑attention on the embedding
    * RBF kernel similarity with a trainable prototype
    * Final fully‑connected classifier
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 embed_dim: int = 4,
                 prototype_dim: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7
        self.embed_proj = nn.Linear(self.flatten_dim, embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.kernel = Kernel()
        self.prototype = nn.Parameter(torch.randn(1, prototype_dim))
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim + embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)

        # Embedding projection
        embed = torch.tanh(self.embed_proj(flat))

        # Self‑attention
        attn_out = self.attention(embed)

        # Kernel similarity with the prototype
        proto = self.prototype.repeat(bsz, 1)
        kernel_sim = self.kernel(embed, proto).unsqueeze(-1)

        # Concatenate all signals
        concat = torch.cat([flat, attn_out, kernel_sim], dim=1)

        out = self.fc(concat)
        return self.norm(out)

__all__ = ["HybridNATModel"]
