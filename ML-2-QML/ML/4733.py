import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Classical self‑attention block that mimics the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor,
                rotation_params: torch.Tensor | None = None,
                entangle_params: torch.Tensor | None = None) -> torch.Tensor:
        # Project to query/key/value spaces
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Build attention scores
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1))
                           / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

class SamplerModule(nn.Module):
    """Small feed‑forward network emulating a quantum sampler."""
    def __init__(self, in_features: int = 2, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Tanh(),
            nn.Linear(4, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class HybridAttentionClassifier(nn.Module):
    """Convolutional backbone + self‑attention + sampler head."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Dropout2d(0.5),
        )
        # Self‑attention
        self.attn = ClassicalSelfAttention(embed_dim=15)
        # Fully‑connected head
        self.fc1 = nn.Linear(15 * 14 * 14, 120)  # assuming 14x14 feature map
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)
        # Sampler output
        self.sampler = SamplerModule(out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Provide dummy parameters if none supplied
        rotation = torch.randn(x.shape[-1], device=x.device)
        entangle = torch.randn(x.shape[-1], device=x.device)
        x = self.attn(x, rotation, entangle)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        # Mix with sampler for richer output
        samp = self.sampler(logits)
        return (probs + samp) / 2

__all__ = ["ClassicalSelfAttention", "SamplerModule", "HybridAttentionClassifier"]
