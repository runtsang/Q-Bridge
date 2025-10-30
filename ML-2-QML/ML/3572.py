import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class RBFKernelLayer(nn.Module):
    """Trainable radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernelFeatureExtractor(nn.Module):
    """Return kernel values between inputs and a fixed support set."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False, n_basis: int = 64):
        super().__init__()
        self.kernel = RBFKernelLayer(gamma, trainable)
        self.n_basis = n_basis
    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(1)          # (batch,1,d)
        support_exp = support.unsqueeze(0)  # (1,n_basis,d)
        k = self.kernel(x_exp, support_exp)
        return k.squeeze(-1)  # (batch, n_basis)

class HybridHead(nn.Module):
    """Dense head that produces a binary probability vector."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

class QuantumKernelHybridNet(nn.Module):
    """CNN backbone + classical RBF kernel + hybrid head."""
    def __init__(self, n_basis: int = 64, shift: float = 0.0):
        super().__init__()
        # Convolutional backbone (identical to the seed)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Classical kernel feature extractor
        self.kernel_extractor = ClassicalKernelFeatureExtractor(gamma=1.0, n_basis=n_basis)
        # Hybrid head
        self.hybrid_head = HybridHead(in_features=n_basis, shift=shift)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # CNN forward
        x = F.relu(self.conv1(inputs))
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
        # Classical kernel mapping using a fixed support set
        support = x[:self.kernel_extractor.n_basis]
        rbf_features = self.kernel_extractor(x, support)
        # Final classification head
        return self.hybrid_head(rbf_features)

__all__ = ["QuantumKernelHybridNet", "RBFKernelLayer", "ClassicalKernelFeatureExtractor", "HybridHead"]
