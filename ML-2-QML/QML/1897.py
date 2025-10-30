"""Quantum-enhanced quanvolution using PennyLane."""
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilterQML(nn.Module):
    """Apply a variational quantum kernel to each 2×2 patch."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        def circuit(patch):
            # Encode patch into rotation angles
            for i in range(n_qubits):
                qml.RY(patch[i], wires=i)
            # Variational layers
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                for i in range(n_qubits):
                    qml.RZ(patch[i] * 0.1, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = qml.qnode(self.dev, interface="torch")(circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(-1, 4)  # (B*14*14, 4)
        # Scale to [0, π/2] for rotation angles
        patches = patches / 255.0 * (torch.pi / 2)
        q_out = self.circuit(patches)  # (B*14*14, n_qubits)
        return q_out.view(B, -1)

class QuanvolutionClassifierQML(nn.Module):
    """Quantum-enhanced classifier with a classical linear head."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQML(n_qubits, n_layers)
        self.linear = nn.Linear(n_qubits * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilterQML", "QuanvolutionClassifierQML"]
