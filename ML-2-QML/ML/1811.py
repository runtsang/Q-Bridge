import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical implementation of the Quanvolution hybrid architecture.
    Uses a three‑layer convolutional front‑end followed by a two‑layer MLP.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        # MLP head
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten and MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridClassifier"]
