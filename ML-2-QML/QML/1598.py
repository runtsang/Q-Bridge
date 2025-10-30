import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np

class QuanvolutionNet(nn.Module):
    """
    Hybrid quanvolution network using a shared variational quantum circuit per 2×2 patch.
    The circuit is parameter‑shared across all patches and optimized end‑to‑end.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_qubits = 4  # one qubit per pixel in a 2×2 patch

        # Quantum device (CPU simulator)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # Shared variational parameters
        self.params = nn.Parameter(torch.randn(num_layers, self.num_qubits, 1))

        # Residual linear conv to match dimensions
        self.res_conv = nn.Conv2d(
            in_channels, self.num_qubits, kernel_size=1, stride=2, bias=False
        )
        self.classifier = nn.Linear(self.num_qubits * 14 * 14, num_classes)

        # QNode for the variational circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            # Encode inputs onto qubits
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RY(params[l, q, 0], wires=q)
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x_img = x.view(bsz, 28, 28)
        patches = []
        # Extract 2×2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x_img[:, r, c],
                        x_img[:, r, c + 1],
                        x_img[:, r + 1, c],
                        x_img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Normalize pixel values to [0, π] for Ry rotations
                patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8) * np.pi
                meas = self.circuit(patch_norm, self.params)
                patches.append(meas)
        out = torch.cat(patches, dim=1)  # shape: (bsz, num_qubits * 14 * 14)
        out = out.view(bsz, self.num_qubits, 14, 14)
        # Residual addition
        res = self.res_conv(x_img)  # shape: (bsz, num_qubits, 14, 14)
        out = out + res
        out = out.view(bsz, -1)
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
