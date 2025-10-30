import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionHybridClassifier(nn.Module):
    """
    Quantum implementation of the Quanvolution hybrid architecture.
    Applies a trainable variational circuit to 2×2 image patches, then
    feeds the concatenated measurements into a two‑layer MLP.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_qubits = 4
        self.n_layers = 2
        # Trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(self.n_layers, self.n_qubits))
        # MLP head
        self.fc1 = nn.Linear(4 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 10)
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Define the variational circuit as a Pennylane QNode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift", batch_mode="vectorized")
        def qnode(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Input encoding: Ry rotations for each pixel
            for i in range(self.n_qubits):
                qml.RY(x[:, i], wires=i)
            # Variational layers
            for layer in range(weights.shape[0]):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Entangling CNOT ring
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Measurement of Pauli‑Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._qnode = qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Extract 2×2 patches across the 28×28 image
        patches = []
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                patch = x[:, :, i:i + 2, j:j + 2]          # shape (batch, 1, 2, 2)
                patch = patch.view(batch_size, -1)          # shape (batch, 4)
                # Apply the quantum kernel
                q_features = self._qnode(patch, self.params)  # shape (batch, 4)
                patches.append(q_features)
        # Concatenate features from all patches
        features = torch.cat(patches, dim=1)  # shape (batch, 4*14*14)
        # Feed into the MLP head
        x = F.relu(self.fc1(features))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridClassifier"]
