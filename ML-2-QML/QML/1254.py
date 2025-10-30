import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionGen200(nn.Module):
    """
    Quantum quanvolution filter that replaces the classical 2×2 conv
    with a variational quantum circuit.  Each 2×2 patch is encoded
    into a 4‑qubit circuit, processed by an ansatz of `num_layers`
    entangling + rotation layers, and the expectation values of
    Pauli‑Z are returned as 4 features per patch.  The flattened
    feature map is fed into a linear classifier.
    """
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 2,
                 num_layers: int = 3,
                 num_classes: int = 10,
                 device: str = "default.qubit",
                 wires: int = 4,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.device = qml.device(device, wires=wires)
        # Trainable parameters of the ansatz
        self.q_params = nn.Parameter(torch.randn(num_layers, wires, 3))
        # Linear head
        self.classifier = nn.Linear(wires * 14 * 14, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.bn = nn.BatchNorm1d(wires * 14 * 14) if use_batchnorm else None

        # Quantum node
        @qml.qnode(self.device, interface="torch")
        def _qnode(patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # patch: (B, 4)
            for i in range(num_layers):
                for j in range(wires):
                    qml.RY(patch[j], wires=j)
                for j in range(wires):
                    qml.CNOT(wires=[j, (j + 1) % wires])
                for j in range(wires):
                    qml.RZ(params[i, j, 0], wires=j)
                    qml.RY(params[i, j, 1], wires=j)
                    qml.RZ(params[i, j, 2], wires=j)
            return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

        self._qnode = _qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: for each 2×2 patch, compute the quantum
        feature vector, concatenate all patches, optionally
        batch‑norm and dropout, and classify.
        """
        bsz = x.size(0)
        # Ensure input shape (B, C, H, W)
        x = x.view(bsz, 1, 28, 28)
        features = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, 0, r: r + self.patch_size, c: c + self.patch_size]
                # Flatten to (B, 4)
                patch = patch.view(bsz, -1)
                # Compute quantum features for each sample in batch
                patch_features = self._qnode(patch, self.q_params)
                # patch_features: (B, 4)
                features.append(patch_features)
        # Concatenate all patch features: (B, 4*14*14)
        features = torch.cat(features, dim=1)
        if self.bn is not None:
            features = self.bn(features)
        if self.dropout is not None:
            features = self.dropout(features)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

__all__ = ["QuanvolutionGen200"]
