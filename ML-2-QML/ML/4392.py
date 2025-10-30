import torch
import torch.nn as nn
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Fast, lightweight self‑attention block that mirrors the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

class HybridFCL(nn.Module):
    """
    Classical hybrid of convolution, fully connected, and self‑attention.
    Mirrors the architecture of the original FCL example but with a
    convolutional backbone inspired by Quantum‑NAT and a classical
    self‑attention head.
    """
    def __init__(self, n_features: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        # Convolutional encoder (Quantum‑NAT style)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )
        # Classical self‑attention
        self.attention = ClassicalSelfAttention(embed_dim=n_features)
        # Final classifier
        self.classifier = nn.Linear(n_features, n_classes)
        self.norm = nn.BatchNorm1d(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.encoder(x)
        flattened = self.flatten(features)
        out = self.fc(flattened)
        out = self.norm(out)
        attn_out = self.attention(out)
        logits = self.classifier(attn_out)
        return logits

    # Compatibility shim – the original FCL used a run method
    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

__all__ = ["HybridFCL"]
