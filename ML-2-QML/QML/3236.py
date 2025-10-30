"""Quantum hybrid filter that mirrors the classical ``HybridConvFilter``.

Implemented using TorchQuantum, the filter encodes each 2×2 image patch into
four qubits, applies a random variational layer, and measures in the Pauli‑Z
basis.  The output is a flattened feature vector suitable for downstream
classifiers.
"""

from __future__ import annotations

import torch
import torchquantum as tq

__all__ = ["HybridConvFilter"]


class HybridConvFilter(tq.QuantumModule):
    """
    Quantum filter operating on 2×2 image patches.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 127.0,
        n_wires: int = 4,
        n_layers: int = 8,
    ) -> None:
        """
        Args:
            kernel_size: Size of the patch to encode.
            stride: Stride for patch extraction.
            threshold: Intensity threshold for π rotation.
            n_wires: Number of qubits per patch.
            n_layers: Number of random operations in the variational layer.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.n_wires = n_wires
        # Encoder: rotate each qubit by π if pixel > threshold, else 0
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_layers, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an image batch and return a flattened feature vector.

        Args:
            x: Input tensor of shape (batch, 1, H, W).

        Returns:
            Tensor of shape (batch, num_patches * n_wires).
        """
        bsz, _, h, w = x.shape
        device = x.device

        # Extract non‑overlapping patches
        patches = (
            x.unfold(2, self.kernel_size, self.stride)
           .unfold(3, self.kernel_size, self.stride)
           .contiguous()
           .view(bsz, -1, self.kernel_size, self.kernel_size)
        )
        # Reshape to (num_patches, n_wires)
        patches_flat = patches.view(-1, self.n_wires)

        # Threshold‑based rotation angles
        theta = torch.where(
            patches_flat > self.threshold,
            torch.full_like(patches_flat, torch.pi),
            torch.zeros_like(patches_flat),
        )

        # Quantum device for all patches
        qdev = tq.QuantumDevice(self.n_wires, bsz=patches_flat.shape[0], device=device)

        # Encode, variational layer, and measurement
        self.encoder(qdev, theta)
        self.q_layer(qdev)
        measurement = self.measure(qdev)

        # Reshape back to (batch, features)
        features = measurement.view(bsz, -1)
        return features
