import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class Quanvolution__gen168(nn.Module):
    """
    Quantum‑classical hybrid filter using a parameterized PennyLane circuit.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_qubits = 4
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        # Trainable ansatz parameters
        self.ansatz_params = nn.Parameter(torch.randn(2, self.num_qubits))
        # Linear classifier mapping flattened patch features to logits
        self.classifier = nn.Linear(4 * (28 // patch_size) ** 2, num_classes)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, params):
            for i in range(2):
                for w in range(self.num_qubits):
                    qml.RY(params[i, w], wires=w)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, :, r:r + self.patch_size, c:c + self.patch_size]
                patch_vals = patch.view(B, -1)
                if self.in_channels == 2:
                    patch_vals = patch_vals[:, :4]
                else:
                    patch_vals = patch_vals[:, :4]
                # Normalize to [-1, 1]
                patch_vals = (patch_vals - 0.5) * 2
                out = self.qnode(patch_vals, self.ansatz_params)
                patches.append(out)
        features = torch.cat(patches, dim=1)  # (B, 4 * N_patches)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen168"]
