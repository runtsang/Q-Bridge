import torch
import torch.nn as nn
import pennylane as qml

class QFCHybridModel(nn.Module):
    """Quantum‑enhanced model: classical encoder + variational quantum circuit."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4, wires: int = 4, dev_name: str = "default.qubit"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.wires = wires
        self.dev = qml.device(dev_name, wires=wires)
        self.params = nn.Parameter(torch.randn(wires))
        self.norm = nn.BatchNorm1d(num_classes)
        self.head = nn.Linear(wires, num_classes)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
            # Angle‑encoding of classical features
            for i in range(min(self.wires, features.shape[0])):
                qml.RY(features[i], wires=i)
            # Variational rotation layer
            for i in range(self.wires):
                qml.RX(params[i], wires=i)
            # Entangling layer
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement in the Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        features = self.encoder(x)
        features = features.view(bsz, -1)
        q_out = torch.stack([self.circuit(self.params, features[i][:self.wires]) for i in range(bsz)])
        out = self.norm(q_out)
        return self.head(out)

__all__ = ["QFCHybridModel"]
