import pennylane as qml
import torch
from torch import nn

class QuantumNATExtendedQML(nn.Module):
    """Hybrid model that replaces the classical convolutional backbone with a
    variational quantum circuit. The circuit encodes a 4‑wire quantum state
    using a depth‑controlled ansatz and measures all qubits. The classical
    head processes the measurement results exactly as in the seed model.

    Parameters
    ----------
    depth : int
        Number of repetitions of the variational block. Higher depth allows
        richer entanglement but increases circuit cost.
    """
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.n_wires = 4
        # Define a simple variational ansatz with Ry and Rz rotations
        def var_block(wires, params):
            qml.RY(params[0], wires=wires[0])
            qml.RZ(params[1], wires=wires[1])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(params[2], wires=wires[2])
            qml.RZ(params[3], wires=wires[3])

        self.circuit = qml.QNode(var_block,
                                 dev=qml.device("default.qubit", wires=self.n_wires),
                                 interface="torch")

        # Classical linear head identical to seed
        self.fc = nn.Sequential(nn.Linear(self.n_wires, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Prepare classical features: average pool over spatial dims to get 16 features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Encode classical features into quantum parameters
        params = pooled[:, :4]  # take first 4 features as parameters
        # Run variational circuit for each batch element
        out_q = self.circuit(params, self.depth)
        # Measurement results are already in torch format
        out = self.fc(out_q)
        return self.norm(out)

__all__ = ["QuantumNATExtendedQML"]
