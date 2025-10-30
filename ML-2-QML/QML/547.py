import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Quantum version of the Quantum‑NAT model using Pennylane.
    The circuit encodes a pooled image into a 4‑qubit variational ansatz
    and outputs four expectation values.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            # x: [batch, n_wires*4]
            for i in range(self.n_wires):
                qml.RY(x[:, i*4], wires=i)
            for i in range(self.n_wires):
                qml.CNOT(wires=[i, (i+1) % self.n_wires])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_wires)]

        self.circuit = circuit
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # 4x4 feature map flattened
        out = self.circuit(pooled)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
