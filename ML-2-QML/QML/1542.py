import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Device used for the variational circuit
DEFAULT_DEVICE = qml.device("default.qubit", wires=2, shots=100)

class QuantumLayer(nn.Module):
    """
    Layer that evaluates a two‑qubit variational circuit.
    Uses PennyLane's automatic differentiation with the
    parameter‑shift rule for analytic gradients.
    """
    def __init__(self, device: qml.Device = DEFAULT_DEVICE) -> None:
        super().__init__()
        self.device = device
        # Define the circuit as a QNode
        self.qnode = qml.QNode(self._circuit, device=self.device,
                               interface="torch", diff_method="parameter-shift")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        # Input encoding: rotate both qubits by the input value
        qml.RY(x, wires=0)
        qml.RY(x, wires=1)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        # Expectation value of Z on the first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a 1‑D tensor of shape (batch,)
        out = self.qnode(x)
        return out

class QuantumHybridBinaryClassifier(nn.Module):
    """
    Hybrid CNN + variational quantum layer classifier.
    Mirrors the classical architecture but replaces the final dense head
    with a quantum layer followed by a dropout‑based calibration head.
    """
    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        # Convolutional feature extractor (identical to the classical version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.2)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum layer
        self.quantum = QuantumLayer(device=DEFAULT_DEVICE)

        # Calibration head (small MLP with dropout)
        self.calibration = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and fully‑connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Quantum layer applied element‑wise
        quantum_out = self.quantum(x.squeeze(-1))
        quantum_out = quantum_out.view(-1, 1)  # shape (batch, 1)

        # Calibration head
        logits = self.calibration(quantum_out)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["QuantumHybridBinaryClassifier"]
