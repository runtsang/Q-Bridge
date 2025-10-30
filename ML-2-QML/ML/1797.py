import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Adds a residual connection to a 2‑D convolutional layer."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)) + x)

class QuantumAwareActivation(nn.Module):
    """Differentiable sigmoid with a learnable shift, mimicking a quantum expectation."""
    def __init__(self, init_shift: float = 0.0):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(init_shift))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridHead(nn.Module):
    """Fully‑connected layer followed by a quantum‑aware activation."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.act    = QuantumAwareActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))

class QCNet(nn.Module):
    """Extended CNN‑ResNet with a quantum‑aware head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.res1 = ResidualBlock(6)
        self.res2 = ResidualBlock(6)
        self.fc1   = nn.Linear(55815, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)
        self.head   = HybridHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = self.head(x)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["ResidualBlock", "QuantumAwareActivation", "HybridHead", "QCNet"]
