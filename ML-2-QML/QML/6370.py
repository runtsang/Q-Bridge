import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dev = qml.device("default.qubit", wires=2)

def variational_circuit(theta, wires=[0, 1]):
    """Simple entangling circuit with two rotation parameters."""
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RX(theta[0], wires=wires[0])
    qml.RZ(theta[1], wires=wires[1])

@qml.qnode(dev, interface="torch")
def quantum_expectation(theta):
    """Return the expectation value of PauliZ on qubit 0."""
    variational_circuit(theta)
    return qml.expval(qml.PauliZ(0))

class QuantumLayer(nn.Module):
    """Hybrid layer that maps a scalar input to a quantum expectation."""
    def __init__(self, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1)
        batch = x.size(0)
        outputs = torch.empty(batch, dtype=torch.double, device=x.device)
        for i in range(batch):
            theta = torch.tensor([x[i, 0].item(), self.shift], dtype=torch.double, device=x.device)
            outputs[i] = quantum_expectation(theta)
        return outputs

class QCNet(nn.Module):
    """Hybrid CNN + variational quantum layer for binary classification."""
    def __init__(self):
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.3)

        # Fully connected head with residual connection
        self.fc1 = nn.Linear(540, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.res_fc = nn.Linear(120, 1)

        # Quantum hybrid head
        self.quantum = QuantumLayer()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and MLP head
        x = torch.flatten(x, 1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)

        # Residual from first hidden layer
        res = self.res_fc(h1)
        out = out + res

        # Quantum hybrid head
        quantum_out = self.quantum(out.unsqueeze(-1))
        prob = torch.sigmoid(quantum_out).unsqueeze(-1)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["QCNet"]
