import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class HybridHead(nn.Module):
    """Hybrid head that replaces the original quantum expectation layer.

    Args:
        in_features: Number of input features (logits).
        bias: Whether to use a bias term.
        temperature: Initial temperature for calibration.
    """
    def __init__(self, in_features: int, bias: bool = True, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)
        self.temperature = Parameter(torch.tensor(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits / self.temperature)
        return probs

class HybridQuantumClassifier(nn.Module):
    """Hybrid quantum‑to‑classical binary classifier.

    The network consists of a 2‑D CNN backbone followed by a
    hybrid head that applies a trainable temperature‑calibrated
    sigmoid.  The head is a drop‑in replacement for the
    original quantum expectation layer.

    Args:
        temperature: Initial temperature for the hybrid head.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(1, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridHead", "HybridQuantumClassifier"]
