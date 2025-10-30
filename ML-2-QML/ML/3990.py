import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch filter that emulates the quanvolution idea."""
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 4 output channels (patches)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch, 1, 28, 28)
        return self.conv(x)  # (batch, 4, 14, 14)

class HybridQuanvolutionNet(nn.Module):
    """
    Classical CNN that mirrors the hybrid quantum architecture:
    - A quanvolution-inspired filter producing 4 feature maps
    - Two convolutional stages, pooling and dropout
    - Three fully connected layers followed by a sigmoid head
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Feature size after conv/pool sequence: 15 channels × 2 × 2 = 60
        self.fc1 = nn.Linear(60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 1, 28, 28)
        x = self.qfilter(inputs)
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
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuanvolutionNet"]
