import pennylane as qml
import torch
import torch.nn as nn

class QFCModel(nn.Module):
    """Variational quantum circuit with classical post‑processing."""
    def __init__(self,
                 n_qubits: int = 4,
                 n_layers: int = 3,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2))
        self.fc = nn.Linear(n_qubits, 4)
        self.norm = nn.BatchNorm1d(4)
        self._build_ansatz()

    def _build_ansatz(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map: encode each input feature into a rotation about Y
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Entangling pattern (cyclic CNOT chain)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Flatten and slice to match the number of qubits
        inp = torch.flatten(x, 1)[:, :self.n_qubits]
        # Scale to [-π, π] for stable rotations
        inp = 2 * torch.pi * (inp - inp.min()) / (inp.max() - inp.min() + 1e-8)
        # Compute circuit outputs for each batch element
        q_out = []
        for i in range(bsz):
            q_out.append(self.circuit(inp[i], self.weights))
        q_out = torch.stack(q_out)
        out = self.fc(q_out)
        return self.norm(out)

__all__ = ["QFCModel"]
