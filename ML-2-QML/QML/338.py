import pennylane as qml
import torch
import torch.nn as nn

# Quantum device with 4 qubits
dev = qml.device("default.qubit", wires=4)

def encode_features(x):
    """Encode a 4‑dim vector into qubits using Ry rotations."""
    for i, val in enumerate(x):
        qml.RY(val, wires=i)

@qml.qnode(dev, interface="torch")
def variational_circuit(x):
    encode_features(x)
    # Parameterized entangling layer
    for i in range(4):
        qml.RZ(x[i], wires=i)
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    # Second layer
    for i in range(4):
        qml.RX(x[i], wires=i)
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuantumNATExtended(nn.Module):
    """Quantum counterpart of the extended model, using a variational circuit."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.encoder = nn.Linear(16, 4)  # map 16‑dim pooled features to 4 qubits
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # x: (B, 1, H, W) -> average pool to 16 features
        bsz = x.shape[0]
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        encoded = self.encoder(pooled)
        # Run each sample through the QNode
        out_q = torch.stack([variational_circuit(sample) for sample in encoded], dim=0)
        out = self.fc(out_q)
        return self.norm(out)

__all__ = ["QuantumNATExtended"]
