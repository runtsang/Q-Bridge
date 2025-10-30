\
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalApproximation(nn.Module):
    """MLP that mimics the quantum expectation layer."""
    def __init__(self, in_features: int, hidden: int = 32, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x + self.shift)


class HybridQCNet(nn.Module):
    """Classical CNN followed by a classical approximation head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout2d(0.3),
        )

        # Infer feature map size after convolutions
        dummy = torch.zeros(1, 3, 32, 32)
        feature_dim = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Produce 4 values that will match the quantum circuit's 4 parameters
            nn.Linear(84, 4),
        )

        self.head = ClassicalApproximation(4, hidden=64, shift=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        probs = self.head(x)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["ClassicalApproximation", "HybridQCNet"]
