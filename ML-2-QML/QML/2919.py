import pennylane as qml
import torch
import torch.nn as nn

class QuantumConvolution(nn.Module):
    """Variational circuit that encodes a 2×2 patch into four qubits."""
    def __init__(self, wires: int = 4, num_layers: int = 3):
        super().__init__()
        self.wires = wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=self.wires)
        self.qnode = qml.qnode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor) -> torch.Tensor:
        # params shape (batch, wires)
        for i in range(self.wires):
            qml.RY(params[:, i], wires=i)
        for _ in range(self.num_layers):
            for i in range(self.wires):
                qml.RY(params[:, i], wires=i)
            for i in range(0, self.wires - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.wires)], dim=1)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """Patch shape (batch, 2, 2) → features shape (batch, 4)."""
        batch = patch.shape[0]
        ops = patch.view(batch, -1).clamp(-1, 1) * torch.pi
        return self.qnode(ops)

class FraudDetectionNet(nn.Module):
    """
    Quantum‑classical hybrid fraud detection model.
    The quantum part performs a variational convolution on 2×2 image patches.
    The classical part aggregates the quantum features and classifies fraud probability.
    """
    def __init__(self, wires: int = 4, num_layers: int = 3):
        super().__init__()
        self.quantum = QuantumConvolution(wires, num_layers)
        self.classifier = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28) grayscale image
        patches: list[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r + 2, c:c + 2]
                patches.append(self.quantum(patch))
        quantum_feat = torch.cat(patches, dim=1)
        logits = self.classifier(quantum_feat)
        return torch.sigmoid(logits)

__all__ = ["FraudDetectionNet"]
