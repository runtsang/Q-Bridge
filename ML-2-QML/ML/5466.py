import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class FraudInspiredLinear(nn.Module):
    """Linear layer with scaling and shift inspired by the photonic fraud example."""
    def __init__(self, in_features: int, out_features: int, params: dict):
        super().__init__()
        weight = torch.tensor([[params['bs_theta'], params['bs_phi']],
                               [params['squeeze_r'][0], params['squeeze_r'][1]]], dtype=torch.float32)
        bias = torch.tensor(params['phases'], dtype=torch.float32)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params['displacement_r'], dtype=torch.float32)
        self.shift = torch.tensor(params['displacement_phi'], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

class HybridBinaryClassifier360(nn.Module):
    """Hybrid classical‑quantum inspired binary classifier."""
    def __init__(self,
                 use_kernel: bool = True,
                 gamma: float = 1.0,
                 fraud_params: dict | None = None):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Optional kernel layer
        self.use_kernel = use_kernel
        if use_kernel:
            self.kernel = RBFKernel(gamma)
            self.register_buffer("support", torch.randn(10, 1))

        # Optional fraud‑inspired linear
        self.fraud_layer = FraudInspiredLinear(2, 2, fraud_params) if fraud_params else None

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

        if self.fraud_layer is not None:
            x = self.fraud_layer(x)

        if self.use_kernel:
            batch = x.size(0)
            k = torch.zeros(batch, self.support.size(0), device=x.device)
            for i in range(batch):
                for j in range(self.support.size(0)):
                    k[i, j] = self.kernel(x[i], self.support[j])
            x = k.mean(dim=1, keepdim=True)

        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier360"]
