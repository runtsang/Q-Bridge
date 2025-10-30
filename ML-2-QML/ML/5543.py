import torch
from torch import nn


class RBFKernel(nn.Module):
    """Radial basis function kernel used as a feature extractor."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumInspiredLayer(nn.Module):
    """A two‑stage linear layer emulating a quantum random layer followed by trainable weights."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.random = nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.normal_(self.random.weight, mean=0.0, std=0.1)
        self.random.weight.requires_grad = False
        self.trainable = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(self.random(x))


class HybridQCNNModel(nn.Module):
    """
    A hybrid convolutional neural network that integrates:
      * Classical convolution‑like fully‑connected layers.
      * A quantum‑inspired random layer.
      * An RBF kernel branch that enriches the representation.
    The design mirrors the QCNN architecture while adding regression‑specific
    components from QuantumRegression and kernel‑based feature extraction.
    """
    def __init__(self, input_dim: int = 8, gamma: float = 1.0):
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.q_layer = QuantumInspiredLayer(16, 16)
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        # Kernel branch
        self.kernel = RBFKernel(gamma)
        self.prototype = nn.Parameter(torch.randn(12))
        # Further processing
        self.conv2 = nn.Sequential(nn.Linear(13, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Head
        self.head = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.q_layer(x)
        x = self.pool1(x)
        # Kernel value with prototype
        k = self.kernel(x, self.prototype)  # shape (batch,1)
        # Concatenate kernel output to features
        x = torch.cat([x, k], dim=-1)  # (batch,13)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Append kernel again before head
        k2 = self.kernel(x, self.prototype)
        x = torch.cat([x, k2], dim=-1)  # (batch,5)
        return self.head(x).squeeze(-1)


__all__ = ["HybridQCNNModel"]
