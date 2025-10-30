import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Classical sampler network that mimics a quantum sampler."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical‑quantum model that fuses quanvolution, QFC, and sampler."""
    def __init__(self, num_classes: int = 10, regression: bool = False):
        super().__init__()
        self.regression = regression
        # Classical convolutional backbone mimicking the quanvolution filter
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.head = nn.Linear(64, 1) if regression else nn.Linear(64, num_classes)
        self.sampler = SamplerQNN(hidden_dim=64)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.bn(self.conv(x))
        flat = features.view(x.size(0), -1)
        hidden = self.fc(flat)
        logits = self.head(hidden)
        probs = self.sampler(hidden)
        if self.regression:
            return logits.squeeze(-1), probs
        return logits, probs

__all__ = ["QuanvolutionHybrid"]
