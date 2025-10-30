"""Hybrid quanvolution network using Pennylane variational circuits.

Each 2×2 image patch is encoded into the angles of a 4‑qubit
variational circuit.  The circuit contains a fixed entangling pattern
and a depth‑controlled number of learnable rotation layers.  All
parameters are stored as PyTorch tensors, allowing end‑to‑end
gradient flow with the classical classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionNet(nn.Module):
    """Variational quanvolution for MNIST‑style images.

    Parameters
    ----------
    in_channels
        Number of input image channels (default 1 for MNIST).
    num_classes
        Number of classification targets.
    depth
        Number of variational layers applied to each patch.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.wires = 4
        # Trainable rotation angles for each layer
        self.theta = nn.Parameter(torch.randn(depth, self.wires))
        # Linear classifier head
        self.linear = nn.Linear(self.wires * 4 * 14 * 14, num_classes)
        # QPU device (default simulator)
        self.dev = qml.device("default.qubit", wires=self.wires)
        # Pre‑define the QNode
        self.qnode = qml.qnode(self.dev, interface="torch")(self._circuit)

    def _circuit(self, x: torch.Tensor) -> [torch.Tensor]:
        """Variational circuit for a single patch.

        Parameters
        ----------
        x
            Tensor of shape (N, 4) containing pixel values for one patch.

        Returns
        -------
        list[torch.Tensor]
            Expectation values of Pauli‑Z on each qubit, shape (N,).
        """
        # Encode pixel values as Ry rotations
        for i in range(self.wires):
            qml.RY(x[:, i], wires=i)
        # Variational layers with learnable angles
        for d in range(self.depth):
            for w in range(self.wires):
                qml.RY(self.theta[d, w], wires=w)
            # Fixed entangling pattern
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
        # Return expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing log‑softmax logits.

        Parameters
        ----------
        x
            Input tensor of shape (B, C, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape (B, num_classes).
        """
        bsz = x.shape[0]
        patches = []
        # Iterate over image patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r : r + 2, c : c + 2].reshape(bsz, -1)
                # Quantum feature vector for batch of patches
                qfeat = torch.stack(self.qnode(patch), dim=1)  # (B, 4)
                patches.append(qfeat)
        # Stack all patch features: (B, 196, 4)
        patches = torch.stack(patches, dim=1)
        flat = patches.reshape(bsz, -1)  # (B, 4*196)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionNet"]
