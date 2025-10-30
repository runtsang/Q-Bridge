"""Quantum component of the hybrid quanvolution model.

Implemented with torchquantum.  Each 2×2 image patch is encoded into
four qubits, processed by a stack of RandomLayers, and measured in the
Pauli‑Z basis.  The module outputs a 4‑dimensional feature vector per
patch, yielding a total feature size of 4×14×14 for a 28×28 input.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from dataclasses import dataclass
from typing import List, Sequence

@dataclass
class FraudLayerParameters:
    """Parameter set used to build a single quantum layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuanvolutionQuantumFilter(tq.QuantumModule):
    """
    Quantum kernel that maps 2×2 image patches to a 4‑dimensional
    measurement vector using a parameterised 4‑qubit circuit.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers

        # Encoder: one Ry gate per pixel
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random layers provide a non‑trivial entangling kernel
        self.random_layers: List[tq.RandomLayer] = [
            tq.RandomLayer(n_ops=8, wires=list(range(n_wires))) for _ in range(n_layers)
        ]

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 28, 28) or (B, 28, 28).
        Returns:
            Tensor of shape (B, 4 * 14 * 14) – 4‑dimensional
            measurement per 2×2 patch.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Ensure 28×28 image format
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = x.view(bsz, 28, 28)

        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch
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
                for layer in self.random_layers:
                    layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

__all__ = ["FraudLayerParameters", "QuanvolutionQuantumFilter"]
