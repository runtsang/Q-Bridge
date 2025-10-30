import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """Quantum‑inspired model using a depth‑controlled variational circuit."""
    def __init__(self, num_qubits: int = 4, depth: int = 2, trainable: bool = True, device: str = "cpu"):
        super().__init__()
        self.trainable = trainable
        self.num_qubits = num_qubits
        self.depth = depth

        # Classical encoder mirroring the feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # Variational parameters
        self.weights = nn.Parameter(torch.randn(self.depth, self.num_qubits, 3))

        # Output layer
        self.fc = nn.Linear(self.num_qubits, 4)
        self.norm = nn.BatchNorm1d(4)

        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode classical data onto qubits
            for i in range(self.num_qubits):
                qml.RY(inputs[:, i], wires=i)
            # Variational layers
            for layer in range(self.depth):
                for q in range(self.num_qubits):
                    qml.RX(self.weights[layer, q, 0], wires=q)
                    qml.RY(self.weights[layer, q, 1], wires=q)
                    qml.RZ(self.weights[layer, q, 2], wires=q)
                # Entangling layer
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])
            # Readout
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from the quantum circuit."""
        x = self.encoder(x)
        x = F.avg_pool2d(x, kernel_size=6).view(x.size(0), -1)
        if x.size(1) > self.num_qubits:
            x = x[:, :self.num_qubits]
        elif x.size(1) < self.num_qubits:
            pad = torch.zeros(x.size(0), self.num_qubits - x.size(1), device=x.device)
            x = torch.cat([x, pad], dim=1)
        q_out = self.circuit(x)
        logits = self.fc(q_out)
        return self.norm(logits)

__all__ = ["QuantumNATEnhanced"]
