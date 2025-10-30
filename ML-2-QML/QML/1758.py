"""Quantum‑classical quanvolution module with trainable quantum kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionEnhanced(tq.QuantumModule):
    """
    Hybrid quanvolution model that applies a trainable quantum kernel to 2×2 image patches
    and combines the result with a classical linear decoder.
    """

    def __init__(self, num_outputs: int = 10, num_classes: bool = True) -> None:
        """
        Args:
            num_outputs: Number of output units (default 10 for MNIST classification).
            num_classes: If True, the final activation will be log_softmax for classification.
                         If False, raw logits are returned for regression or other tasks.
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.num_classes = num_classes

        # Quantum kernel parameters
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable random layer – each gate is parameterized and learned
        self.q_layer = tq.RandomLayer(
            n_ops=8, wires=list(range(self.n_wires)), fixed=False
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear decoder
        self.classifier = nn.Linear(4 * 14 * 14, self.num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid quanvolution network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Log‑softmax logits if num_classes is True, otherwise raw logits.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to 28x28 per image
        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # 2x2 patch flattened to 4 values
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode the patch into the quantum device
                self.encoder(qdev, data)
                # Apply trainable random layer
                self.q_layer(qdev)
                # Measure all qubits
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        # Concatenate all patch measurements into a single feature vector
        features = torch.cat(patches, dim=1)  # shape: (bsz, 4*14*14)

        # Decode with classical linear layer
        logits = self.classifier(features)

        if self.num_classes:
            return F.log_softmax(logits, dim=-1)
        return logits


__all__ = ["QuanvolutionEnhanced"]
