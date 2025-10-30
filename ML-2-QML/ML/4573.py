"""Classical binary classifier with thresholded sigmoid head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThresholdedSigmoid(nn.Module):
    """Sigmoid with optional shift and threshold."""
    def __init__(self, shift: float = 0.0, threshold: float = 0.0):
        super().__init__()
        self.shift = shift
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift - self.threshold)

class HybridBinaryNet(nn.Module):
    """Classical CNN classifier with a thresholded sigmoid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = ThresholdedSigmoid(shift=0.0, threshold=0.0)

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

__all__ = ["HybridBinaryNet"]
