"""Quantum implementation of the hybrid quanvolution classifier.

A torchquantum quantum filter processes 2×2 image patches to
extract entangled features.  The resulting vector is fed into a
lightweight sampler network (MLP) that maps to class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionHybridClassifier(tq.QuantumModule):
    """
    Quantum quanvolution classifier with a sampler head.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    num_classes : int, default 10
        Number of target classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4

        # Quantum encoder: rotate each qubit by an input‑dependent angle
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random variational layer to entangle the qubits
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical sampler head (MLP) that maps quantum features to logits
        # The head is lightweight to keep the quantum contribution dominant.
        self.sampler_head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Prepare 2×2 patches from the 28×28 image
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        # Concatenate all patch measurements into a feature vector
        features = torch.cat(patches, dim=1)

        # Pass quantum features through the sampler head
        logits = self.sampler_head(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
