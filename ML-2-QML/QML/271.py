import torch
import pennylane as qml
import pennylane.numpy as pnp
import torch.nn as nn
from torch import Tensor

class Quanvolution__gen301(nn.Module):
    """
    Quantum quanvolution module based on PennyLane.
    Encodes 2×2 image patches into a 4‑qubit circuit, applies a parameterized
    variational layer, and measures all qubits. The output is concatenated
    across the spatial grid and returned as a tensor of shape (batch, 4*14*14).
    """

    def __init__(self, dev_name: str = "default.qubit", wires: int = 4, num_layers: int = 2):
        super().__init__()
        self.dev = qml.device(dev_name, wires=wires)
        # Initialize circuit parameters
        self.params = nn.Parameter(
            torch.randn(num_layers, wires, dtype=torch.float64)
        )

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _qnode(patch: Tensor, params: Tensor) -> Tensor:
            # Encode each pixel with a Ry rotation
            for i in range(wires):
                qml.RY(patch[..., i], wires=i)
            # Entangling layer
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Parameterized rotations
            for layer in range(params.shape[0]):
                for i in range(wires):
                    qml.RY(params[layer, i], wires=i)
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measure all qubits in Z basis
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1)), \
                   qml.expval(qml.PauliZ(wires=2)), qml.expval(qml.PauliZ(wires=3))

        self._qnode = _qnode

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, 28, 28).

        Returns:
            Tensor of shape (batch, 4*14*14) containing quantum‑encoded features.
        """
        bsz = x.size(0)
        patches = []
        # Extract 2x2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]
                # Flatten patch to shape (batch, 4)
                patch = patch.view(bsz, -1)
                # Run quantum circuit
                out = self._qnode(patch, self.params)
                # out shape (batch, 4)
                patches.append(out)
        # Concatenate all patch outputs
        features = torch.cat(patches, dim=1)
        return features

__all__ = ["Quanvolution__gen301"]
