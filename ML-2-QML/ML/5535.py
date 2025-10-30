import numpy as np
import torch
from torch import nn
from typing import Sequence, Iterable

class RBFKernel(nn.Module):
    """Purely classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class FullyConnectedLayer(nn.Module):
    """Simple fully connected layer with tanh activation."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().cpu().numpy()

class BinaryClassifier(nn.Module):
    """Fully classical binary classifier mirroring QCNet architecture."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

class QuantumKernelHybrid(nn.Module):
    """Composable hybrid model combining kernel, filter, fully‑connected layer and classifier."""
    def __init__(self):
        super().__init__()
        self.kernel = RBFKernel()
        self.filter = QuanvolutionFilter()
        self.fcl = FullyConnectedLayer()
        self.classifier = BinaryClassifier()

    def forward(self, data: torch.Tensor, params: Iterable[float]) -> torch.Tensor:
        features = self.filter(data)
        kernel_val = self.kernel(features, features)
        logits = torch.sigmoid(torch.tensor(self.fcl(params), dtype=torch.float32))
        return logits, kernel_val

__all__ = ["RBFKernel", "kernel_matrix", "QuanvolutionFilter",
           "FullyConnectedLayer", "BinaryClassifier", "QuantumKernelHybrid"]
