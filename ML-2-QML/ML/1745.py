import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Conv2d, ReLU, MaxPool2d, Linear, Dropout

class QuantumNATEnhanced(nn.Module):
    """Enhanced hybrid classical model for multi‑output regression.

    Extends the original Quantum‑NAT CNN with depth‑wise separable convolutions,
    gated residual connections, and a multi‑head pooling strategy.  The final
    fully‑connected head outputs four features and normalises them with a
    BatchNorm1d layer.  The module also exposes a ``predict`` helper for
    quick inference and a ``set_device`` method to move parameters to a chosen
    backend device.
    """

    def __init__(self) -> None:
        super().__init__()
        # Depth‑wise separable conv block 1
        self.dw_conv1 = nn.Sequential(
            Conv2d(1, 8, kernel_size=3, stride=1, padding=1, groups=1),
            ReLU(),
            MaxPool2d(2),
        )
        # Point‑wise conv to increase channel dim
        self.pt_conv1 = nn.Sequential(
            Conv2d(8, 16, kernel_size=1, stride=1, padding=0),
            ReLU(),
        )
        # Residual gate
        self.res_gate = nn.Sequential(
            Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        # Second depth‑wise separable conv block
        self.dw_conv2 = nn.Sequential(
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1, groups=32),
            ReLU(),
            MaxPool2d(2),
        )
        self.pt_conv2 = nn.Sequential(
            Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            ReLU(),
        )
        # Multi‑head pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Fully connected head
        self.fc = nn.Sequential(
            Linear(64 * 2, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 4),
        )
        self.norm = BatchNorm1d(4)

    def set_device(self, device: torch.device) -> "QuantumNATEnhanced":
        """Move model parameters to the specified device."""
        return self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual gated shortcut
        residual = self.res_gate(x)
        x = self.dw_conv1(x)
        x = self.pt_conv1(x)
        x = x + residual
        x = self.dw_conv2(x)
        x = self.pt_conv2(x)
        # Multi‑head pooling
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        pooled = torch.cat([avg, max_], dim=1)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc(pooled)
        return self.norm(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience inference wrapper."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

__all__ = ["QuantumNATEnhanced"]
