import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleSelfAttention(nn.Module):
    """Learnable self‑attention over a single feature vector."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        scores = torch.softmax(q @ k.t() / np.sqrt(x.size(-1)), dim=-1)
        return scores @ v  # shape (batch, embed_dim)

class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel used for prototype similarity."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridAttentionKernelNet(nn.Module):
    """
    Classical convolutional network with prototype‑based self‑attention and RBF kernel aggregation.
    """
    def __init__(self,
                 n_prototypes: int = 10,
                 prototype_dim: int = 84,
                 gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, prototype_dim)

        # Prototype bank – learnable
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prototype_dim))

        # Kernel and attention modules
        self.kernel = RBFKernel(gamma)
        self.attention = SimpleSelfAttention(prototype_dim)

        # Final classification head
        self.classifier = nn.Linear(prototype_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        feat = self.fc2(x)  # (batch, embed_dim)

        # Self‑attention scores between feature and prototypes
        att_weights = self.attention(feat)  # (batch, embed_dim)

        # Kernel similarities between feature and each prototype
        kernel_sims = torch.stack([self.kernel(feat, p) for p in self.prototypes], dim=1)  # (batch, n_prototypes)

        # Combine attention and kernel signals
        combined = torch.softmax(kernel_sims + att_weights.sum(dim=1, keepdim=True), dim=1)

        # Weighted prototype aggregation
        aggregated = (combined.unsqueeze(-1) * self.prototypes).sum(dim=1)  # (batch, embed_dim)

        # Final classification head
        logits = self.classifier(aggregated).squeeze(-1)  # (batch,)
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridAttentionKernelNet"]
