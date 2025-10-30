import torch
import torch.nn as nn
import pennylane as qml

class HybridQFCModel(nn.Module):
    """Hybrid quantum‑classical model: classical pooling, quantum ansatz, and batch‑norm."""
    def __init__(self, n_wires: int = 4, n_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_wires, shots=None)
        self.weights = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(self.n_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            qml.AngleEmbedding(inputs, wires=range(self.n_wires))
            for layer in range(self.n_layers):
                for wire in range(self.n_wires):
                    qml.RX(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                    qml.RZ(weights[layer, wire, 2], wires=wire)
                for wire in range(self.n_wires):
                    qml.CNOT(wires=[wire, (wire + 1) % self.n_wires])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = torch.nn.functional.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        q_out = self.qnode(pooled, self.weights)
        q_out = torch.stack(q_out, dim=1)
        q_out = self.dropout(q_out)
        return self.norm(q_out)

__all__ = ["HybridQFCModel"]
