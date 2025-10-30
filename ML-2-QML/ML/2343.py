import torch
from torch import nn
import numpy as np
from typing import Sequence

class ClassicalKernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz with a learnable gamma parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`ClassicalKernalAnsatz` exposing a matrix interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class QCNNHybrid(nn.Module):
    """Hybrid classical‑quantum convolutional network.

    The architecture mimics a QCNN with a classical convolution‑like backbone
    and a quantum kernel module that can be swapped out or frozen.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # Classical convolution‑style layers
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

        # Classical kernel module
        self.kernel = Kernel(gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the internal classical kernel."""
        return self.kernel.matrix(a, b)

def QCNNHybridFactory(gamma: float = 1.0) -> QCNNHybrid:
    """Factory returning a configured :class:`QCNNHybrid` instance."""
    return QCNNHybrid(gamma)

__all__ = ["ClassicalKernalAnsatz", "Kernel", "QCNNHybrid", "QCNNHybridFactory"]
