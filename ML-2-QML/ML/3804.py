import torch
import torch.nn as nn

class Conv(nn.Module):
    """
    Classical convolutional network that emulates the original quanvolution filter
    and extends it with a fully‑connected head.  The architecture mirrors the
    Quantum‑NAT style while remaining purely classical.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv1 = nn.Conv2d(1, 8, kernel_size=kernel_size, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)
        # Compute flattened size after convolutions
        dummy = torch.zeros(1, 1, 28, 28)
        out = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(dummy))))))
        self.flat_size = out.view(1, -1).size(1)
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.fc2 = nn.Linear(64, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.sigmoid(x - self.threshold)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x - self.threshold)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return self.norm(x)

__all__ = ["Conv"]
