import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Hybrid neural network using a variational quantum circuit as the convolutional filter
class QuanvolutionEnhanced(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        # 4 qubits for a 2×2 image patch
        self.dev = qml.device("default.qubit", wires=4, shots=None)
        # Variational parameters for the quantum circuit
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers * 4))
        # Linear classifier mapping 4×14×14 features to 10 classes
        self.linear = nn.Linear(4 * 14 * 14, 10)

        # Variational quantum circuit (qnode)
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encode the 2×2 patch into Ry rotations
            for i, wire in enumerate(range(4)):
                qml.RY(inputs[i], wires=wire)
            # Variational layers
            for l in range(n_layers):
                for i, wire in enumerate(range(4)):
                    qml.RY(weights[l * 4 + i], wires=wire)
                # Entangling gates
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])
            # Measurement: expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reshape to (bsz, 28, 28)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch (bsz, 2, 2)
                patch = x[:, r:r+2, c:c+2]
                patch = patch.view(bsz, 4)  # flatten to (bsz, 4)
                # Apply the quantum circuit to each sample in the batch
                out = []
                for i in range(bsz):
                    out.append(self.circuit(patch[i], self.weights))
                out = torch.stack(out, dim=0)  # shape: (bsz, 4)
                patches.append(out)
        # Concatenate all patches: shape (bsz, 4*14*14)
        patches = torch.cat(patches, dim=1)
        logits = self.linear(patches)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionEnhanced"]
