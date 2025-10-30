"""Quantum‑augmented quanvolution with dual‑branch fusion.

This module implements the same high‑level interface as the
classical implementation but replaces the quantum‑style branch with
a real variational quantum circuit.  The circuit processes 2×2
image patches on a 4‑qubit device and measures the expectation
value of the Pauli‑Z operator on each qubit.  The device is
wrapped in a PennyLane QNode, which allows end‑to‑end differentiable
training when used with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionFusion(nn.Module):
    """Hybrid classical‑quantum fusion for MNIST‑style data.

    The architecture mirrors :class:`QuanvolutionFusion` from the
    classical module but uses a genuine quantum circuit for the
    quantum branch.  The gate parameter ``alpha`` controls the fusion
    of the two branches as before.
    """

    def __init__(self) -> None:
        super().__init__()
        # Classical branch (identical to the classical module)
        self.classical_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # Quantum circuit parameters
        self.n_wires = 4
        self.device = qml.device("default.qubit", wires=self.n_wires)
        # Trainable parameters for a 3‑layer entangling ansatz
        self.theta = nn.Parameter(torch.randn(3, self.n_wires))

        # Gate
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Final classifier
        self.classifier = nn.Linear(4 * 14 * 14, 10)

        # QNode
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs, theta):
            # Encode inputs into qubits using Ry rotations
            for i in range(self.n_wires):
                qml.RY(inputs[i], wires=i)
            # Entangling layers
            for layer in range(theta.shape[0]):
                for i in range(self.n_wires):
                    qml.CNOT(wires=[i, (i + 1) % self.n_wires])
                for i in range(self.n_wires):
                    qml.RY(theta[layer, i], wires=i)
            # Measure expectation of Pauli‑Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        cls_feat = self.classical_conv(x)      # (B, 4, 14, 14)
        cls_flat = cls_feat.view(x.size(0), -1)

        # Quantum branch
        # Extract patches (B, 14, 14, 4)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        patches = patches.contiguous().view(x.size(0), 14, 14, 4)
        B, H, W, _ = patches.shape
        # Run each patch through the quantum circuit
        q_out = []
        for i in range(H):
            for j in range(W):
                inp = patches[:, i, j, :]  # (B, 4)
                # Circuit expects inputs in [-pi, pi] range; rescale
                inp = inp * (torch.pi / 2)
                out = self.circuit(inp, self.theta)  # (B, 4)
                q_out.append(out)
        # Stack and reshape back to (B, 4, 14, 14)
        q_feat = torch.stack(q_out, dim=1)            # (B, H*W, 4)
        q_feat = q_feat.permute(0, 2, 1).view(B, 4, H, W)
        q_flat = q_feat.view(x.size(0), -1)

        # Fuse
        alpha = torch.sigmoid(self.alpha)
        feat = alpha * cls_flat + (1 - alpha) * q_flat

        logits = self.classifier(feat)
        return F.log_softmax(logits, dim=-1)

    @staticmethod
    def contrastive_loss(cls_feats: torch.Tensor,
                         q_feats: torch.Tensor,
                         temperature: float = 0.5
                         ) -> torch.Tensor:
        """Same contrastive loss as in the classical module."""
        cls_norm = F.normalize(cls_feats, dim=1)
        q_norm = F.normalize(q_feats, dim=1)
        sim = cls_norm @ q_norm.t() / temperature
        labels = torch.arange(cls_feats.size(0), device=cls_feats.device)
        loss = F.cross_entropy(sim, labels)
        return loss
