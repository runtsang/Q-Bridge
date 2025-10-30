import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumHybridBinaryClassifier(nn.Module):
    """Hybrid model that uses a quantum expectation value as the logit."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, 120)

        # Quantum head
        self.device = qml.device("default.qubit", wires=2)
        self.qnode = qml.QNode(self._quantum_circuit, self.device, interface="torch", diff_method="parameter-shift")
        self.out = nn.Linear(1, num_classes)

    def _quantum_circuit(self, params):
        """Two‑qubit variational circuit."""
        for i, w in enumerate(params):
            qml.RY(w, wires=i)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        params = x[:, :2]  # first two features as rotation angles
        quantum_out = self.qnode(params)
        quantum_out = quantum_out.unsqueeze(-1)  # shape (batch, 1)
        logits = self.out(quantum_out)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridBinaryClassifier"]
