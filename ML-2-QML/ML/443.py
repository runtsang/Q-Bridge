import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """ML head with batch‑norm for stabilising training."""
    def __init__(self, in_features: int, hidden: int = 64, out_features: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.bn1(h)
        h = F.relu(self.fc2(h))
        h = self.bn2(h)
        return self.out(h)

class QuantumHybridBinaryClassifier(nn.Module):
    """Hybrid model that uses a classical residual MLP head for binary classification."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, 120)  # assumes 32×32 input images
        self.residual_head = ResidualMLP(120, hidden=64, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        logits = self.residual_head(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ResidualMLP", "QuantumHybridBinaryClassifier"]
