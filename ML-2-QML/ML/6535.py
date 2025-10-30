import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Dropout, BatchNorm2d, Sequential
import pennylane as qml

class ResBlock(nn.Module):
    """A lightweight residual block for the convolutional stem."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = BatchNorm2d(out_ch)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(out_ch)
        self.downsample = (
            nn.Sequential(Conv2d(in_ch, out_ch, kernel_size=1, stride=stride))
            if stride!= 1 or in_ch!= out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class HybridLayer(nn.Module):
    """Differentiable quantum-to-torch conversion using Pennylane."""
    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit", shots: int = 1024):
        super().__init__()
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.n_qubits = n_qubits
        self.qnode = qml.qnode(self.dev, interface='torch', diff_method='backprop')(self._circuit)

    def _circuit(self, theta: torch.Tensor):
        """Quantum ansatz: RY rotations followed by CNOT chain."""
        for i in range(self.n_qubits):
            qml.RY(theta[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, n_qubits)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        outputs = []
        for i in range(inputs.shape[0]):
            outputs.append(self.qnode(inputs[i]))
        return torch.stack(outputs).unsqueeze(-1)

class QuantumHybridClassifier(nn.Module):
    """Extended hybrid network with residual CNN and quantum head."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.stem = Sequential(
            Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=1),
            ResBlock(6, 6),
            Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=1),
            ResBlock(15, 15)
        )
        self.fc1 = Linear(540, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 1)
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.5)
        self.hybrid = HybridLayer(n_qubits=1, dev_name="default.qubit", shots=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.hybrid(x)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
