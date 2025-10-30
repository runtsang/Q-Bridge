"""Quantum variant of the QuanvolutionSamplerNet.

This network replaces the classical sampler MLP with a
parameter‑driven variational circuit that emulates the
SamplerQNN.  Each 2×2 image patch is encoded into four qubits via
Ry gates, entangled with CX links, and subject to trainable
Ry rotations.  The measurement of each qubit provides a 4‑dimensional
feature vector identical in shape to the classical MLP output.
The final linear classifier mirrors the classical counterpart.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionSamplerNet(tq.QuantumModule):
    """Quantum implementation of the quanvolution filter with a sampler circuit.

    The architecture mirrors the classical version:
        1. 2×2 patch extraction via convolution (implemented in the
           forward loop for convenience)
        2. Each patch → 4‑qubit quantum circuit (Ry encoding,
           CX entanglement, Ry weights)
        3. Measurement of all qubits → 4‑dimensional feature vector
        4. Linear classifier
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder: one Ry per pixel
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable rotation weights (4 per qubit)
        self.weights = nn.Parameter(torch.randn(self.n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classifier head
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=device)

        # Pre‑process image into 2×2 patches
        x = x.view(batch, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Shape (B, 4) patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Reset device for each patch
                qdev.reset()
                # Encode pixel values into qubits
                self.encoder(qdev, patch)
                # Entangle qubits (CX chain)
                qdev.cx(0, 1)
                qdev.cx(1, 2)
                qdev.cx(2, 3)
                # Apply trainable rotations
                for w, wire in zip(self.weights, range(self.n_wires)):
                    qdev.ry(w, wire)
                # Measure all qubits
                meas = self.measure(qdev)
                # Convert expectation values to probabilities
                probs = (meas + 1) / 2  # (B, 4)
                patches.append(probs)

        # Concatenate all patch outputs (B, 4*14*14)
        output = torch.cat(patches, dim=1)
        logits = self.classifier(output)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionSamplerNet"]
