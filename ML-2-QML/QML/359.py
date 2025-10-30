import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pennylane import numpy as np

class PennylaneCircuit(nn.Module):
    """
    Variational circuit with depth‑controlled layers and a learnable basis rotation.
    """
    def __init__(self, num_qubits: int, depth: int, device: str = "default.qubit") -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        self.params = nn.Parameter(torch.randn(depth, num_qubits, 3))
        self.basis = nn.Parameter(torch.randn(num_qubits, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, num_qubits)
        batch_size = inputs.shape[0]
        expectations = []
        for i in range(batch_size):
            def circuit(*params):
                # basis rotation
                for w, (rx, ry, rz) in enumerate(zip(self.basis, self.basis, self.basis)):
                    qml.RX(rx, wires=w)
                    qml.RY(ry, wires=w)
                    qml.RZ(rz, wires=w)
                # variational layers
                for layer in range(self.depth):
                    for w in range(self.num_qubits):
                        qml.RX(params[layer, w, 0], wires=w)
                        qml.RY(params[layer, w, 1], wires=w)
                        qml.RZ(params[layer, w, 2], wires=w)
                    # entangling
                    for w in range(self.num_qubits - 1):
                        qml.CNOT(wires=[w, w + 1])
                return qml.expval(qml.PauliZ(0))
            circuit = qml.QNode(circuit, self.dev, interface="torch")
            expectations.append(circuit(*self.params))
        return torch.stack(expectations)

class QuantumHybridBinaryClassifier(nn.Module):
    """
    Hybrid CNN followed by a Pennylane variational circuit head.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 15 * 15, 120)  # adjust according to input size
        self.fc2 = nn.Linear(120, 84)
        self.quantum_head = PennylaneCircuit(num_qubits, depth)
        self.fc3 = nn.Linear(num_qubits, 1)  # map quantum output to logits

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
        # quantum head expects batch_size x num_qubits
        quantum_out = self.quantum_head(x[:, :self.quantum_head.num_qubits])
        logits = self.fc3(quantum_out)
        probs = torch.sigmoid(logits).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QuantumHybridBinaryClassifier"]
