import torch
import torch.nn as nn
import torch.nn.functional as F

# The quantum layer is defined in the QML module
from.QuantumHybridBinaryClassifier_qml import QuantumHybridLayer


class ResidualDenseBlock(nn.Module):
    """A lightweight residual block that keeps the spatial resolution unchanged."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (nn.Identity()
                         if in_channels == out_channels
                         else nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=1, padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class QuantumHybridBinaryClassifier(nn.Module):
    """Full hybrid classifier combining a residual‑dense CNN with a 3‑qubit quantum head."""
    def __init__(self,
                 n_qubits: int = 3,
                 shift: float = 3.1415926535 / 2,
                 shots: int = 1024) -> None:
        super().__init__()
        # Classical feature extractor
        self.resblock1 = ResidualDenseBlock(3, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.resblock2 = ResidualDenseBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected head
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)  # Output a single logit
        # Quantum layer
        self.quantum = QuantumHybridLayer(n_qubits=n_qubits,
                                          shift=shift,
                                          shots=shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 3, H, W]
        x = self.resblock1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape: [batch, 1]
        # Expand to match quantum input shape
        quantum_input = x.squeeze(-1)  # shape: [batch]
        quantum_output = self.quantum(quantum_input)  # shape: [batch, 1]
        probs = torch.sigmoid(quantum_output)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["ResidualDenseBlock", "QuantumHybridBinaryClassifier"]
