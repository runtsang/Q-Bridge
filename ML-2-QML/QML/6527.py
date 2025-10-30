import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class Quanvolution__gen364(nn.Module):
    """
    Quantum-enhanced quanvolution filter with a parameterized variational ansatz.
    Each 2x2 image patch is encoded via Ry gates and processed by a depth‑controlled
    circuit of Ry and CNOT layers. The expectation values of Pauli‑Z are used as features.
    """
    def __init__(self, depth: int = 2, num_classes: int = 10):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.dev = qml.device('default.qubit', wires=4)

        def circuit(patch, weights):
            for i in range(4):
                qml.RY(patch[i], wires=i)
            for d in range(self.depth):
                for i in range(4):
                    qml.RY(weights[d, i], wires=i)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]

        weight_shapes = {"weights": (self.depth, 4)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.linear = nn.Linear(4 * 14 * 14, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        patches = []
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                patch = x[:, :, i:i+2, j:j+2]
                patch_flat = patch.view(bsz, 4)
                out = self.qlayer(patch_flat)
                patches.append(out)
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen364"]
