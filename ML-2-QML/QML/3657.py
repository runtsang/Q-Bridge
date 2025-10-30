"""Quantum quanvolution filter and classifier using torchquantum.

This module implements a variational circuit that processes 2×2 image patches
with a parameterized rotation followed by a random layer and Pauli‑Z measurement.
The output of each patch is concatenated and fed into a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum filter + classifier.  Mirrors the classical counterpart for direct
    comparison of performance and scalability.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits per patch.  Must be 4 for 2×2 patches.  Default ``4``.
    n_layers : int, optional
        Number of random layers applied after encoding.  Default ``8``.
    threshold : float, optional
        Input threshold used for encoding.  Values above threshold are encoded
        as π rotations, below as 0.  Default ``0.5``.
    num_filters : int, optional
        Number of output features per patch.  Default ``4``.
    """

    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 8,
        threshold: float = 0.5,
        num_filters: int = 4,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.threshold = threshold
        self.num_filters = num_filters

        # Encoder maps a 2×2 patch to qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Random variational layer
        self.q_layer = tq.RandomLayer(n_ops=n_layers, wires=list(range(self.n_wires)))
        # Measurement of all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Linear classifier head
        self.classifier = nn.Linear(num_filters * 14 * 14, 10)

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
        bsz = x.shape[0]
        device = x.device

        # Prepare quantum device for batch processing
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch and encode
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Threshold‑based encoding: values > threshold -> π, else 0
                encoded = torch.where(data > self.threshold, torch.pi, torch.zeros_like(data))
                self.encoder(qdev, encoded)
                self.q_layer(qdev)
                measurement = self.measure(qdev)  # (bsz, n_wires)
                # Convert PauliZ output (-1, 1) to 0/1 representation
                features = (measurement + 1) / 2
                patches.append(features)

        # Concatenate patch features: (B, num_filters*14*14)
        features = torch.cat(patches, dim=1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
