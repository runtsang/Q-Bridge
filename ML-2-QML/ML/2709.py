import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Hybrid classical sampler network that mirrors the quantum SamplerQNN
    but augments it with a lightweight CNN encoder and batch‑normalised
    output.  The architecture is inspired by the Quantum‑NAT convolutional
    front‑end and the classical SamplerQNN's two‑layer MLP."""
    def __init__(self) -> None:
        super().__init__()
        # Encoder: one‑channel 8‑feature map, 3×3 conv, ReLU, 2×2 pool
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten then fully‑connected head
        self.head = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over two classes."""
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        logits = self.head(flattened)
        out = self.norm(logits)
        return F.softmax(out, dim=-1)

__all__ = ["SamplerQNN"]
