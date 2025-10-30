import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np

# Quantum device
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(params: torch.Tensor) -> torch.Tensor:
    """Single‑qubit rotation followed by Pauli‑Z expectation."""
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper around the quantum circuit implementing the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        # Compute expectation for each input in the batch
        return torch.stack([quantum_circuit(inp) for inp in inputs])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        shift = np.pi / 2
        grad = torch.zeros_like(inputs)
        for i, val in enumerate(inputs):
            right = quantum_circuit(val + shift)
            left = quantum_circuit(val - shift)
            grad[i] = (right - left) / (2 * shift)
        return grad * grad_output

class Hybrid(nn.Module):
    """Layer that maps a scalar feature to a quantum‑derived probability."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x)

class ResidualBlock(nn.Module):
    """Same residual block as in the classical implementation."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class HybridBinaryClassifier(nn.Module):
    """Hybrid CNN + quantum expectation head for binary classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)
        self.hybrid = Hybrid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Quantum expectation head returns values in [-1,1]; map to [0,1] via (x+1)/2
        quantum_out = (self.hybrid(x).squeeze(-1) + 1) / 2
        probs = torch.stack([quantum_out, 1 - quantum_out], dim=-1)
        return probs
