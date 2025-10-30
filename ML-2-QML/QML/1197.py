import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Define a parameterized quantum circuit
def pqc(params, wires):
    """A simple variational circuit with rotation and entangling layers."""
    for i, wire in enumerate(wires):
        qml.RY(params[0, i], wires=wire)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    for i, wire in enumerate(wires):
        qml.RZ(params[1, i], wires=wire)

# Quantum device with batched execution
dev = qml.device("default.qubit", wires=4, shots=None)

# QNode with interface torch
@qml.qnode(dev, interface="torch", diff_method="backprop", batch_mode=True)
def quantum_circuit(x, params):
    # x: (batch, 4)
    for i in range(4):
        qml.RY(x[:, i], wires=i)
    # Apply variational circuit
    pqc(params, wires=range(4))
    # Return expectation values of PauliZ on each qubit
    return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(4)], dim=1)

class QuanvolutionFilter(nn.Module):
    """Quantum quanvolution filter using a parameterized circuit."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        # Learnable parameters for the variational circuit: 2 layers * 4 qubits
        self.params = nn.Parameter(torch.randn(2, 4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reshape to (batch, 28, 28)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2x2 patch and flatten
                patch = x[:, r:r+2, c:c+2].reshape(bsz, -1)
                # Run quantum circuit
                out = quantum_circuit(patch, self.params)
                patches.append(out)
        # Concatenate all patch outputs: shape (batch, 4 * 14 * 14)
        out = torch.cat(patches, dim=1)
        out = self.dropout(out)
        return out

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using the quantum quanvolution filter followed by a small MLP head."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.mlp = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
