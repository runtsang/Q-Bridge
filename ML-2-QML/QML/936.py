import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import math

class QuantumNATEnhanced(nn.Module):
    """Quantum neural network with a variational circuit and a learnable encoder."""
    def __init__(self,
                 n_qubits: int = 4,
                 init_depth: int = 3,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = init_depth
        self.dev = qml.device(device, wires=n_qubits)

        # Encoder that maps 16 classical features to the number of qubits
        self.encoder = nn.Linear(16, n_qubits)

        # Trainable weights for the variational layers: depth × n_qubits × 2
        self.weights = nn.Parameter(torch.randn(self.depth, n_qubits, 2))

        # Quantum node with parameter‑shift gradient estimation
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            # Angle encoding of the classical data
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            for d in range(self.depth):
                for i in range(n_qubits):
                    qml.RX(weights[d, i, 0], wires=i)
                    qml.RZ(weights[d, i, 1], wires=i)
                # Entangling layer (full‑chain + wrap‑around)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qml.QNode(circuit,
                               self.dev,
                               interface="torch",
                               diff_method="parameter-shift")

        # Classical head mapping the quantum expectation values to the target
        self.fc = nn.Linear(n_qubits, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encoder → quantum circuit → linear head.
        Expected input shape: (batch, 1, 28, 28)
        """
        bsz = x.shape[0]
        # Classical preprocessing: average‑pool to 16 features per sample
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # shape: (batch, 16)
        # Encode to qubit space
        encoded = self.encoder(pooled)  # shape: (batch, n_qubits)
        # Scale to [-π, π] for angle encoding
        encoded = 2 * math.pi * encoded / (encoded.max() + 1e-8)

        # Run the quantum circuit for each sample in the batch
        qout = []
        for i in range(bsz):
            qout.append(self.qnode(encoded[i], self.weights))
        qout = torch.stack(qout)  # shape: (batch, n_qubits)

        # Classical post‑processing
        out = self.fc(qout)
        return self.bn(out)

__all__ = ["QuantumNATEnhanced"]
