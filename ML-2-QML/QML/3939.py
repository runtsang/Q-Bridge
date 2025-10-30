"""QuanvolutionGraphQL: Quantum + Graph hybrid for image classification.

This module defines a single public class, QuanvolutionGraphQLClassifier, that
mirrors the classical implementation but replaces the patch extractor with a
variational quantum circuit and uses the measurement results as node features.
The graph adjacency is built from fidelity (cosine similarity) of the
measurement vectors.  The rest of the pipeline is identical to the classical
version.

The class can be used as a drop‑in replacement for the classical version,
allowing experimentation with quantum kernels while keeping the same
training interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np


class QuanvolutionGraphQLClassifier(nn.Module):
    """Hybrid quantum+graph class for image classification.

    Attributes
    ----------
    encoder : tq.GeneralEncoder
        Encodes a 2×2 image patch into a 4‑qubit state.
    q_layer : tq.RandomLayer
        Variational layer applied to the qubits.
    measure : tq.MeasureAll
        Measures all qubits in the Pauli‑Z basis.
    linear : nn.Linear
        Final linear classifier.
    threshold : float
        Fidelity threshold for graph edges.
    secondary : float | None
        Lower similarity threshold.
    secondary_weight : float
        Weight of secondary edges.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 2,
        stride: int = 2,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def _quantum_patch(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> torch.Tensor:
        """Encode, evolve, and measure a 2×2 patch on a quantum device.

        Parameters
        ----------
        qdev : tq.QuantumDevice
            Quantum device with 4 qubits.
        patch : torch.Tensor
            Shape (B, 4) – pixel values of the 2×2 patch.

        Returns
        -------
        torch.Tensor
            Measurement outcomes of shape (B, 4) with values in {−1, +1}.
        """
        self.encoder(qdev, patch)
        self.q_layer(qdev)
        measurement = self.measure(qdev)
        return measurement

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2×2 patches and run them through the quantum device.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Concatenated measurement vectors of shape (B, 196, 4).
        """
        B, C, H, W = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=B, device=x.device)
        patches = []
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                data = torch.stack(
                    [
                        x[:, 0, r, c],
                        x[:, 0, r, c + 1],
                        x[:, 0, r + 1, c],
                        x[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                meas = self._quantum_patch(qdev, data)
                patches.append(meas)
        patches = torch.stack(patches, dim=1)  # (B, 196, 4)
        return patches

    def _build_adjacency(self, patches: torch.Tensor) -> torch.Tensor:
        """Build a weighted adjacency matrix from patch measurement vectors.

        Parameters
        ----------
        patches : torch.Tensor
            Shape (B, N, D) where N=196, D=4.

        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (N, N) with weights 1.0 for edges
            above ``threshold`` and ``secondary_weight`` for edges above
            ``secondary`` (if provided).
        """
        patch_vecs = patches[0]  # (N, D)
        similarity = torch.einsum("nd,md->nm", patch_vecs, patch_vecs)
        similarity = similarity / (
            patch_vecs.norm(p=2, dim=-1, keepdim=True)
            * patch_vecs.norm(p=2, dim=-1, keepdim=True).transpose(-1, -2)
            + 1e-12
        )
        adjacency = torch.ones_like(similarity)
        adjacency[similarity < self.threshold] = 0.0
        if self.secondary is not None:
            mask_secondary = (similarity >= self.secondary) & (similarity < self.threshold)
            adjacency[mask_secondary] = self.secondary_weight
        return adjacency

    def _aggregate_with_graph(self, patches: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Aggregate patch features via weighted adjacency."""
        aggregated = torch.einsum("nm,bmd->bnd", adjacency, patches)
        return aggregated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, num_classes).
        """
        patches = self._extract_patches(x)  # (B, 196, 4)
        adjacency = self._build_adjacency(patches)  # (196, 196)
        aggregated = self._aggregate_with_graph(patches, adjacency)  # (B, 196, 4)
        flattened = aggregated.view(aggregated.size(0), -1)  # (B, 784)
        logits = self.linear(flattened)
        return F.log_softmax(logits, dim=-1)
