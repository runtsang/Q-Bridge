"""QuanvolutionHybrid with a parameterized quantum circuit for feature extraction.

The class shares the same public API as the classical version but offers a
hybrid quantum‑classical branch that applies a variational quantum circuit to
each 2×2 patch of the input image. The quantum circuit is parameterized by
learnable rotation angles and uses PennyLane's default simulator. The
classical branch remains available for comparison.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionHybrid(nn.Module):
    """Hybrid quanvolution model with learnable patch extractor and variational quantum circuit.

    Parameters
    ----------
    patch_size : int, default 2
        Size of the square patch extracted by the convolution.
    stride : int, default 2
        Stride of the convolution, controlling the number of patches.
    num_classes : int, default 10
        Number of output classes.
    mode : str, default 'hybrid'
        Default mode for the forward pass; must be either 'classical' or
        'hybrid'.
    """
    def __init__(
        self,
        patch_size: int = 2,
        stride: int = 2,
        num_classes: int = 10,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        self._mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.num_classes = num_classes

        # Learnable patch extractor: 1 input channel → 4 output channels
        self.patch_extractor = nn.Conv2d(
            1, 4, kernel_size=patch_size, stride=stride
        )

        # Linear classifier
        num_patches = (28 // stride) ** 2
        self.linear = nn.Linear(4 * num_patches, num_classes)

        # Quantum circuit parameters
        self.n_wires = 4
        self.n_params = 8  # number of variational parameters per layer
        self.q_params = nn.Parameter(torch.randn(self.n_params))

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(self.dev, interface="torch")
        def qnode(patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            """Apply variational circuit to a single 2×2 patch and return a 4‑dimensional feature vector."""
            # Encode the patch into qubit states
            for i in range(self.n_wires):
                qml.RY(patch[i], wires=i)
            # Parameterized rotation layer
            for i in range(self.n_params):
                qml.RX(params[i], wires=i % self.n_wires)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)])

        self.qnode = qnode

    def forward(self, x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).
        mode : str, optional
            Override the instance mode for this call.  Must be
            'classical' or 'hybrid'.

        Returns
        -------
        torch.Tensor
            Log‑softmax of logits.
        """
        if mode is None:
            mode = self._mode

        if mode == "classical":
            patches = self.patch_extractor(x)  # (bsz, 4, H', W')
            features = patches.view(patches.size(0), -1)  # flatten
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)

        elif mode == "hybrid":
            # Extract patches as flattened vectors
            patches = self.patch_extractor(x)  # shape (bsz, 4, H', W')
            bsz, _, Hp, Wp = patches.shape
            # Prepare a list to collect quantum features
            quantum_features = []
            # Iterate over each patch location
            for i in range(Hp * Wp):
                # Extract the i‑th patch across the batch: shape (bsz, 4)
                patch = patches[:, :, i]
                # Apply quantum circuit to each sample in the batch
                qfeat_batch = []
                for j in range(bsz):
                    qfeat = self.qnode(patch[j], self.q_params)  # (4,)
                    qfeat_batch.append(qfeat)
                qfeat_batch = torch.stack(qfeat_batch, dim=0)  # (bsz, 4)
                quantum_features.append(qfeat_batch)
            # Stack all patch features: (bsz, Hp*Wp, 4)
            quantum_features = torch.stack(quantum_features, dim=1)
            # Flatten to match the linear head input: (bsz, 4 * Hp * Wp)
            features = quantum_features.view(bsz, -1)
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

__all__ = ["QuanvolutionHybrid"]
