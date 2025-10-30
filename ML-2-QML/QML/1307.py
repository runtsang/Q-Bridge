import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

class QuanvolutionHybrid(nn.Module):
    """Quantum‑inspired quanvolution filter backed by a variational circuit.

    The circuit encodes a 2×2 image patch into a 4‑qubit register using Ry
    rotations, applies two variational layers of Ry and CNOTs, and measures
    each qubit in the Z basis.  The resulting 4‑dimensional feature vector
    is concatenated across the 14×14 patches and fed to a classical linear
    head.  An optional rotation‑angle head is provided to demonstrate
    multi‑task learning.
    """
    def __init__(self, num_classes: int = 10, use_rotation_head: bool = False,
                 device: str = "default.qubit", wires: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.use_rotation_head = use_rotation_head
        self.wires = wires
        self.n_layers = n_layers

        # Parameterized variational parameters
        self.variational_params = nn.Parameter(
            torch.randn(n_layers, wires, dtype=torch.float32)
        )

        # Define the quantum device
        self.q_dev = qml.device(device, wires=wires)

        # Define the QNode
        self.qnode = qml.QNode(self._circuit, self.q_dev)

        # Classifier
        self.classifier = nn.Linear(wires * 14 * 14, num_classes)

        if use_rotation_head:
            self.rotation_head = nn.Linear(wires * 14 * 14, 4)

    def _circuit(self, patch: np.ndarray, params: np.ndarray):
        # Encode pixel values
        for i in range(self.wires):
            qml.Ry(patch[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.wires):
                qml.Ry(params[layer, i], wires=i)
            # Entanglement
            for i in range(self.wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.wires - 1, 0])

        # Measure in Z basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        bsz = x.size(0)
        features_list = []

        # Convert to numpy for PennyLane
        x_np = x.cpu().numpy()

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2x2 patch: shape (B, 4)
                patch = np.stack([
                    x_np[:, r, c],
                    x_np[:, r, c + 1],
                    x_np[:, r + 1, c],
                    x_np[:, r + 1, c + 1]
                ], axis=1)
                batch_features = []
                for i in range(bsz):
                    feat = self.qnode(patch[i], self.variational_params.detach().cpu().numpy())
                    batch_features.append(feat)
                features_list.append(torch.tensor(batch_features, device=x.device, dtype=torch.float32))

        # Concatenate all patches: (B, 4*14*14)
        features = torch.cat(features_list, dim=1)

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def predict_rotation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation_head:
            raise RuntimeError("Rotation head not enabled.")
        features = self.feature_extractor(x)
        logits = self.rotation_head(features)
        return F.log_softmax(logits, dim=-1)

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        # Same code as forward but without final classifier
        bsz = x.size(0)
        features_list = []
        x_np = x.cpu().numpy()
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = np.stack([
                    x_np[:, r, c],
                    x_np[:, r, c + 1],
                    x_np[:, r + 1, c],
                    x_np[:, r + 1, c + 1]
                ], axis=1)
                batch_features = []
                for i in range(bsz):
                    feat = self.qnode(patch[i], self.variational_params.detach().cpu().numpy())
                    batch_features.append(feat)
                features_list.append(torch.tensor(batch_features, device=x.device, dtype=torch.float32))
        return torch.cat(features_list, dim=1)
