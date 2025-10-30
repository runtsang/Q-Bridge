import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np

class QuanvolutionGen256(nn.Module):
    """Hybrid quanvolution network that replaces the classical filter with a 16‑qubit quantum kernel.

    Each 16×16 patch is encoded into a 16‑qubit state via Ry rotations.  A trainable 4‑layer circuit
    with entangling CNOTs produces a 16‑dim expectation vector.  A linear layer expands this to
    a 256‑dim patch representation, followed by the same residual attention and classification head
    as the classical counterpart.
    """

    def __init__(self, num_classes: int = 10, patch_size: int = 16, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.qubits_per_patch = 16  # number of qubits used per patch

        # Pad image to 32×32
        self.pad = nn.ZeroPad2d((0, 4, 0, 4))

        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.qubits_per_patch, shots=0)

        # Trainable rotation parameters for the quantum circuit
        self.theta = nn.Parameter(torch.randn(self.qubits_per_patch))

        # Build the qnode
        self._build_qnode()

        # Linear mapping from 16‑dim quantum features to 256‑dim patch embedding
        self.patch_proj = nn.Linear(self.qubits_per_patch, embed_dim)

        # Residual connection
        self.residual = nn.Identity()

        # Self‑attention across four patches
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Linear(embed_dim * 4, num_classes)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(patch: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            """Quantum circuit that encodes pixel values and applies trainable rotations."""
            # Encode pixel intensities into Ry rotations
            for i in range(self.qubits_per_patch):
                qml.RY(patch[:, i], wires=i)

            # Trainable rotation layer
            for i in range(self.qubits_per_patch):
                qml.RY(theta[:, i], wires=i)

            # Entangling layers (4 repetitions)
            for _ in range(4):
                for i in range(self.qubits_per_patch - 1):
                    qml.CNOT(i, i + 1)
                for i in range(self.qubits_per_patch - 1):
                    qml.CNOT(i + 1, i)

            # Expectation values of PauliZ on each qubit
            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.qubits_per_patch)]
            return torch.stack(expvals, dim=-1)

        self.qnode = qnode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape (B, num_classes).
        """
        B = x.size(0)

        # Pad and extract 16×16 patches
        x = self.pad(x)  # (B, 1, 32, 32)
        patches = torch.nn.functional.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # (B, patch_size*patch_size, 4)
        patches = patches.permute(0, 2, 1)  # (B, 4, 256)

        # Reshape for quantum evaluation
        patches_flat = patches.reshape(B * 4, self.patch_size * self.patch_size)  # (B*4, 256)

        # Use only the first 16 qubits of each patch for the circuit
        pixel_vals = (patches_flat[:, :self.qubits_per_patch] / 255.0) * np.pi  # (B*4, 16)

        # Broadcast trainable parameters to match batch size
        theta = self.theta.unsqueeze(0).repeat(B * 4, 1)  # (B*4, 16)

        # Run quantum circuit
        qfeat = self.qnode(pixel_vals, theta)  # (B*4, 16)

        # Map quantum features to 256‑dim patch embedding
        emb = self.patch_proj(qfeat)  # (B*4, 256)

        # Reshape back to (B, 4, 256)
        emb = emb.reshape(B, 4, self.embed_dim)

        # Residual
        emb = emb + self.residual(emb)

        # Self‑attention
        attn_out, _ = self.attn(emb, emb, emb)  # (B, 4, embed_dim)

        # Flatten and classify
        flat = attn_out.reshape(B, -1)  # (B, 4*embed_dim)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen256"]
