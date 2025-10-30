import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class VariationalCircuit(nn.Module):
    """Two‑qubit variational circuit with a trainable rotation angle."""
    def __init__(self, dev: qml.Device, shots: int = 1000) -> None:
        super().__init__()
        self.dev = dev
        self.shots = shots
        self.theta = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.RY(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the variational circuit."""
    def __init__(self, dev: qml.Device, shots: int = 1000, shift: float = np.pi/2) -> None:
        super().__init__()
        self.circuit = VariationalCircuit(dev, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.squeeze(-1).unsqueeze(-1)  # shape (batch, 1, 1)
        return self.circuit(batch)

class QCNet(nn.Module):
    """CNN followed by a variational quantum head."""
    def __init__(self, dev: qml.Device, shots: int = 1000, shift: float = np.pi/2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(dev, shots, shift)

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
        # Quantum head expects a batch of angles
        q_out = self.hybrid(x)
        return torch.cat((q_out, 1 - q_out), dim=-1)

__all__ = ["VariationalCircuit", "Hybrid", "QCNet"]
