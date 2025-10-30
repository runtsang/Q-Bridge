import torch
import torch.nn as nn
import torch.nn.functional as F
from.QuantumKernelMethod import Kernel
from.Conv import Conv as ConvFilter


class QCNNGenML(nn.Module):
    """Hybrid classical network that mimics a QCNN using a quantum‑inspired kernel head."""

    def __init__(self, num_prototypes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # CNN backbone identical to QCNet
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum‑inspired kernel head
        self.kernel = Kernel(gamma=gamma)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1))

        # Optional lightweight conv‑filter to emulate quanvolution
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
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
        x = self.fc3(x).squeeze(-1)

        # Kernel head: compute RBF similarity to trainable prototypes
        # x shape: [batch]
        # prototypes shape: [num_prototypes, 1]
        # Resulting kernel matrix shape: [batch, num_prototypes]
        kernel_vals = torch.exp(
            -self.kernel.gamma
            * (x.unsqueeze(1) - self.prototypes.t()) ** 2
        )
        probs = torch.sigmoid(kernel_vals.sum(dim=1))
        return torch.stack([probs, 1 - probs], dim=-1)


def QCNN() -> QCNNGenML:
    """Factory returning the configured classical QCNN‑style model."""
    return QCNNGenML()


__all__ = ["QCNN", "QCNNGenML"]
