"""QuantumHybridClassifier – classical dense head with gated fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridClassifier(nn.Module):
    """
    Classical implementation of the hybrid classifier.
    The model consists of a convolutional backbone identical to the
    original quantum version, followed by a gated hybrid head that
    blends two independent dense layers.
    """
    def __init__(self, in_features: int = 1, shift: float = 0.1) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head: two parallel dense layers
        self.head_a = nn.Linear(self.fc3.out_features, 1)
        self.head_b = nn.Linear(self.fc3.out_features, 1)
        self.shift = shift
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid head
        logits_a = self.head_a(x)
        logits_b = self.head_b(x)
        gate = torch.sigmoid(self.shift * logits_a)
        logits = gate * logits_a + (1 - gate) * logits_b
        probs = self.sigmoid(logits).view(-1, 1)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridClassifier"]
