import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridHead(nn.Module):
    """Classical surrogate for a quantum expectation head."""
    def __init__(self, in_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, hidden)
        self.tanh = nn.Tanh()
        self.mean_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.tanh(out)
        out = self.mean_pool(out.unsqueeze(-1)).squeeze(-1)
        return out

class ConvBackbone(nn.Module):
    """CNN backbone inspired by Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        return x

class HybridQuantumClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier with a classical surrogate head."""
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        self.head = HybridHead(in_features=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumClassifier"]
