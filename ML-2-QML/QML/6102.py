"""Quantum quanvolution network with a trainable variational layer.

The network partitions the input image into 2×2 patches, encodes each
patch into a 4‑qubit quantum state, applies a parameterised random
layer, measures all qubits, and concatenates the results.  A linear
head produces class logits.  The circuit is fully trainable via
gradient descent using TorchQuantum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class Quanvolution(tq.QuantumModule):
    """Quantum quanvolution block with a parameterised circuit."""

    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each pixel as a rotation about the Y axis.
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer
        self.var_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear classifier
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        B = x.shape[0]
        device = x.device
        # Reshape to (B, 28, 28)
        x = x.view(B, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # (B, 4)
                # Create a quantum device for this batch
                qdev = tq.QuantumDevice(self.n_wires, bsz=B, device=device)
                # Encode the patch data
                self.encoder(qdev, patch)
                # Apply the variational layer
                self.var_layer(qdev)
                # Measure all qubits
                measurement = self.measure(qdev)  # (B, 4)
                patches.append(measurement)
        # Concatenate all patch measurements
        features = torch.cat(patches, dim=1)  # (B, 4 * 14 * 14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
