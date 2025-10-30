import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Adds a residual connection to a 2‑D convolutional layer."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)) + x)

class HybridQuantumLayer(nn.Module):
    """Variational layer that forwards activations through a Pennylane circuit."""
    def __init__(self, shift: float = 0.0, shots: int = 100):
        super().__init__()
        self.shift = shift
        self.shots = shots
        self.device = qml.device("default.qubit", wires=2)

        @qml.qnode(self.device, interface="torch", diff_method="backprop",
                   shots=self.shots, batch=True)
        def circuit(params):
            # params shape: (batch, 2)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.RY(params[:, 0], wires=0)
            qml.RY(params[:, 1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1)
        params = torch.cat([x, x], dim=1)  # duplicate for two qubits
        return self.circuit(params).unsqueeze(-1)

class QCNet(nn.Module):
    """CNN backbone followed by a variational Pennylane head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.res1 = ResidualBlock(6)
        self.res2 = ResidualBlock(6)
        self.fc1   = nn.Linear(55815, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)
        self.hybrid = HybridQuantumLayer(shift=0.0, shots=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQuantumLayer", "QCNet"]
