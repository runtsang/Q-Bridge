import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid quantum network that augments the classical input with a
    parameterised variational circuit.

    Key features:
    1. A 4‑qubit PennyLane device with a shallow variational layer.
    2. Encodes a 16‑dimensional pooled feature vector into rotation angles.
    3. Employs a depth‑controlled schedule to mitigate barren plateaus.
    4. Returns a 4‑dimensional feature vector matched to the classical head.
    """

    def __init__(self,
                 n_wires: int = 4,
                 depth: int = 2,
                 shots: int = 1024):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.device = qml.device("default.qubit", wires=n_wires, shots=shots)

        # Encoder: linear mapping from 16‑dim pooled features to qubit angles
        self.encoder = nn.Linear(16, n_wires)

        # Variational parameters: shape (depth, n_wires, 3) for RX,RZ,RY
        self.var_params = nn.Parameter(torch.randn(depth, n_wires, 3))

        self.norm = nn.BatchNorm1d(n_wires)

        # Build the QNode capturing the circuit
        def _circuit(x):
            # Encode the pooled features into rotations
            angles = self.encoder(x)  # (bs, n_wires)
            for i in range(n_wires):
                qml.RX(angles[..., i], wires=i)
            # Variational layers
            for d in range(depth):
                for i in range(n_wires):
                    qml.RX(self.var_params[d, i, 0], wires=i)
                    qml.RY(self.var_params[d, i, 1], wires=i)
                    qml.RZ(self.var_params[d, i, 2], wires=i)
                # Entangle all qubits in a ring topology
                for i in range(n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_wires - 1, 0])
            # Measurement of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        self.circuit = qml.qnode(self.device, interface="torch")(_circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: average‑pool the image to 16 features,
        run the variational circuit, and apply batch‑norm.
        """
        bsz = x.size(0)
        # 2‑D average pooling to produce a 16‑dim vector per sample
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)  # (bs,16)
        out = self.circuit(pooled)                            # (bs, n_wires)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
