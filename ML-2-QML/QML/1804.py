import torch
import torch.nn as nn
import pennylane as qml

class QuantumNATHybrid(nn.Module):
    """
    Hybrid model that replaces the MLP head of the classical variant with
    a parameterized quantum circuit implemented with Pennylane.
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.n_wires = n_wires
        # Feature extractor identical to the classical version
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Feature mixing to 4‑dim vector
        self.mix = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_wires),  # 4‑dim embedding
            nn.BatchNorm1d(n_wires),
            nn.Dropout(dropout)
        )
        # Quantum device and circuit
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Encode inputs as rotations
            for i in range(n_wires):
                qml.RY(inputs[:, i], wires=i)
            # Parameterized entangling layers
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_wires))
            # Measure expectation values of PauliZ
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])

        self.quantum_circuit = circuit
        # Quantum circuit parameters
        weight_shapes = {"weights": (n_layers, n_wires, 3)}
        self.q_params = nn.Parameter(torch.randn(weight_shapes["weights"]))
        # Classical classifier head on top of quantum outputs
        self.classifier = nn.Sequential(
            nn.Linear(n_wires, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        mixed = self.mix(flat)
        # Quantum forward
        q_out = self.quantum_circuit(mixed, self.q_params)
        logits = self.classifier(q_out)
        return logits

__all__ = ["QuantumNATHybrid"]
