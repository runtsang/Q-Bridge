import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum device with 4 wires for the 2×2 patches
dev = qml.device("default.qubit", wires=4)

def _qnode(patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Variational circuit applied to a single 2×2 patch."""
    # Encode the four pixel values into rotations
    for i in range(4):
        qml.RY(patch[i], wires=i)
    # Parameterised layers
    for d in range(params.shape[0]):
        for i in range(4):
            qml.RY(params[d, i], wires=i)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
    return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(4)])

class QuantumLayer(nn.Module):
    """Quantum layer that processes all patches in a batch."""
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth
        # Shared parameters across all patches
        self.params = nn.Parameter(torch.randn(depth, 4))
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Tensor of shape (batch, 4) containing flattened 2×2 pixel values.
        Returns:
            Tensor of shape (batch, 4) with measurement results.
        """
        batch_size = patch.shape[0]
        out = []
        for i in range(batch_size):
            res = _qnode(patch[i], self.params)
            out.append(res)
        return torch.stack(out, dim=0)

class QuanvolutionFilter(nn.Module):
    """Quantum quanvolution filter that applies a variational circuit to each 2×2 patch."""
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.quantum_layer = QuantumLayer(depth=depth)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28).
        Returns:
            Tensor of shape (batch, 4 * 14 * 14) containing flattened quantum features.
        """
        batch_size = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]  # (batch, 2, 2)
                patch_flat = patch.view(batch_size, 4)
                out = self.quantum_layer(patch_flat)
                patches.append(out)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quantum quanvolutional filter followed by a linear head."""
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(depth=depth)
        self.linear = nn.Linear(4 * 14 * 14, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
