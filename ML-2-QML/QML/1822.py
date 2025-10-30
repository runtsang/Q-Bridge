"""Quantum‑aware version of the extended quanvolution module.

The module uses torchquantum to construct a depth‑wise quantum kernel
for each 2×2 image patch.  It follows the same public API as the
classical implementation, providing a ``pretrain_contrastive`` method
that returns the quantum‑filtered features.

Features
--------
* `depth` – number of random layers per patch; increases the expressive
  power of the quantum kernel.
* `dropout` – dropout applied to the concatenated feature map.
* `shortcut` – 1×1 classical convolution that bypasses the quantum
  transform, enabling a residual connection.

The quantum filter is built with a small GeneralEncoder that maps each
pixel of the patch to a qubit via a Ry rotation.  A RandomLayer applies
a configurable number of two‑qubit gates.  The final measurement is
done in the Pauli‑Z basis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑aware depth‑wise quanvolution module.

    Parameters
    ----------
    depth : int, default=2
        Number of random two‑qubit layers applied per patch.
    dropout : float, default=0.1
        Dropout probability applied to the concatenated feature map
        before the linear head.
    """

    def __init__(self, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.depth = depth
        self.n_wires = 4

        # Encoder: map each pixel in the 2×2 patch to a Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random layer with a number of ops proportional to depth
        self.q_layer = tq.RandomLayer(
            n_ops=8 * depth, wires=list(range(self.n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical shortcut to match channel dimension
        self.shortcut = nn.Conv2d(
            in_channels=1,
            out_channels=4 * depth,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(4 * depth * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum filter and linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log softmax logits over 10 classes.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Extract 2×2 patches
        patches = []
        x_reshape = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x_reshape[:, r, c],
                        x_reshape[:, r, c + 1],
                        x_reshape[:, r + 1, c],
                        x_reshape[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # shape (batch, 4)
                # Encode each pixel into a qubit
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patches: shape (batch, 4*14*14)
        quantum_features = torch.cat(patches, dim=1)

        # Shortcut branch
        shortcut_features = self.shortcut(x).view(bsz, -1)

        # Combine
        features = quantum_features + shortcut_features
        features = self.dropout(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def pretrain_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the quantum‑filtered features before the linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, 4*depth*14*14).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        patches = []
        x_reshape = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x_reshape[:, r, c],
                        x_reshape[:, r, c + 1],
                        x_reshape[:, r + 1, c],
                        x_reshape[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        quantum_features = torch.cat(patches, dim=1)
        return quantum_features

__all__ = ["QuanvolutionHybrid"]
