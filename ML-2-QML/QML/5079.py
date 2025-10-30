"""Fully quantum implementation of the QuanvolutionHybrid model.

The class mirrors the classical API but implements a 2‑qubit kernel over image
patches and a linear head for classification.  It can be swapped into the
training pipeline in place of the classical hybrid without any API changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

__all__ = ["QuanvolutionHybrid"]


class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum‑only quanvolution filter with a fully‑connected head.

    The circuit processes 2×2 patches of a 28×28 image using a random
    two‑qubit kernel and measures all qubits in the Pauli‑Z basis.  The
    resulting feature vector is fed into a linear head that produces log‑softmax
    logits for classification.
    """

    def __init__(self, n_wires: int = 4, n_classes: int = 10):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps each pixel value to a rotation about Y
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  Tensor of shape (batch, 28, 28) representing grayscale images.
        Returns:
            Tensor of shape (batch, n_classes) containing log‑softmax logits.
        """
        batch = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=device)

        # Extract 2×2 patches and run the quantum kernel on each
        patch_meas = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                patch_meas.append(self.measure(qdev))

        # Concatenate all patch measurements to form the feature vector
        features = torch.cat(patch_meas, dim=1)  # shape: (batch, n_wires * 14 * 14)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)
