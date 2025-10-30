import torch
from torch import nn
from typing import Optional

class ResidualBlock(nn.Module):
    """A simple residual block that preserves dimensionality."""
    def __init__(self, features: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.fc = nn.Linear(features, features)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x) + x)

class QCNNHybrid(nn.Module):
    """
    Hybrid classical–quantum network that combines a residual QCNN‑style
    convolutional backbone with an optional quantum kernel layer.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden: int = 16,
        quantum_kernel: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden // 2, hidden // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden // 2, hidden // 4), nn.Tanh())
        self.res_block = ResidualBlock(hidden // 4)
        self.quantum_kernel = quantum_kernel
        head_in = hidden // 4 + (1 if quantum_kernel else 0)
        self.head = nn.Linear(head_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.res_block(x)

        if self.quantum_kernel is not None:
            qfeat = self.quantum_kernel(x).unsqueeze(-1)
            x = torch.cat([x, qfeat], dim=-1)

        return torch.sigmoid(self.head(x))

def QCNNHybridFactory(
    in_features: int = 8,
    hidden: int = 16,
    quantum_kernel: Optional[nn.Module] = None,
) -> QCNNHybrid:
    """Convenient factory to instantiate the hybrid model."""
    return QCNNHybrid(in_features, hidden, quantum_kernel)

__all__ = ["QCNNHybrid", "QCNNHybridFactory", "ResidualBlock"]
