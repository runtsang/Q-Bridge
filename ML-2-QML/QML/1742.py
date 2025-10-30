import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen(nn.Module):
    """Variational quanvolution filter with shared parameters across all 2×2 patches."""
    def __init__(self, num_qubits=4, num_layers=2, num_units=10):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)
        # Variational parameters (shared across patches)
        self.params = nn.Parameter(torch.randn(self.num_qubits, self.num_layers))
        # Linear classifier
        self.fc = nn.Linear(num_qubits * 14 * 14, num_units)

    def prep(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values to the range [0, π] for angle encoding."""
        return torch.pi * (x.float() / 255.0)

    def _quantum_block(self, patch: torch.Tensor) -> torch.Tensor:
        """Apply a parameter‑shared variational circuit to a single 2×2 patch."""
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(p):
            # Angle encoding
            for i, val in enumerate(p):
                qml.RY(val, wires=i)
            # Variational layers
            for l in range(self.num_layers):
                for w in range(self.num_qubits):
                    qml.RZ(self.params[w, l], wires=w)
                for w in range(self.num_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            # Return expectation values of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return circuit(patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        bsz, _, h, w = x.shape
        patches = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = x[:, r:r+2, c:c+2]  # shape [bsz, 2, 2]
                patch = patch.view(bsz, -1)  # shape [bsz, 4]
                measurements = torch.stack([self._quantum_block(patch[i]) for i in range(bsz)])
                patches.append(measurements)
        features = torch.cat(patches, dim=1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen"]
