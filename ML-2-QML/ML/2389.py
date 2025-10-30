import numpy as np
import torch
from torch import nn
from typing import Sequence

class ClassicalKernalAnsatz(nn.Module):
    """RBF kernel ansatz using a gamma parameter."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Wraps the classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def classical_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class ClassicalQCNNModel(nn.Module):
    """A classical network mimicking the QCNN structure."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridKernelQCNN(nn.Module):
    """Hybrid classical-quantum kernel + QCNN model."""
    def __init__(self, gamma: float = 1.0, use_quantum: bool = False):
        super().__init__()
        self.gamma = gamma
        self.use_quantum = use_quantum
        self.classical_kernel = ClassicalKernel(gamma)
        self.quantum_kernel = None  # placeholder for a quantum kernel instance
        self.qcnn = ClassicalQCNNModel()

    def set_quantum_kernel(self, quantum_kernel):
        """Attach a quantum kernel implementation."""
        self.quantum_kernel = quantum_kernel
        self.use_quantum = True

    def compute_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        if self.use_quantum and self.quantum_kernel is not None:
            return self.quantum_kernel.kernel_matrix(a, b)
        else:
            return classical_kernel_matrix(a, b, self.gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.qcnn(inputs)

__all__ = ["HybridKernelQCNN", "classical_kernel_matrix", "ClassicalKernalAnsatz", "ClassicalKernel", "ClassicalQCNNModel"]
