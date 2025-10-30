import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerNet(nn.Module):
    """Classical sampler that maps 2‑D inputs to a 4‑D softmax output."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridClassifier(nn.Module):
    """Convolutional backbone followed by a differentiable quantum‑expectation head."""
    def __init__(self, shift: float = math.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.shift = shift

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
        # Replace the quantum expectation with a sigmoid for the classical version
        probs = torch.sigmoid(x + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

class UnifiedSamplerHybridNet(nn.Module):
    """Combines a classical sampler and a hybrid quantum‑classical classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.sampler = SamplerNet()
        self.classifier = HybridClassifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2) – raw sampler input
        sample = self.sampler(x)          # (batch, 4)
        # Expand sample to a 3‑channel 32×32 image for the classifier
        img = sample.unsqueeze(1).repeat(1, 3, 1, 1)  # (batch, 3, 4, 4)
        img = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)
        return self.classifier(img)

__all__ = ["UnifiedSamplerHybridNet"]
