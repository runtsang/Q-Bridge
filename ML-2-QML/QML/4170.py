"""
Hybrid Quanvolution model with quantum autoencoding.

The quantum implementation mirrors the classical pipeline:
  * A variational quanvolution filter that processes 2×2 patches.
  * A per‑patch quantum autoencoder that compresses 4‑qubit states to 2 qubits.
  * Aggregation of the compressed patches and a classical linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchquantum as tq
from typing import Tuple


# --------------------------------------------------------------------------- #
# Quantum autoencoder – variational encoder + decoder
# --------------------------------------------------------------------------- #

class QuantumAutoencoder(tq.QuantumModule):
    """
    Variational autoencoder that compresses a 4‑qubit state to 2 qubits.

    The encoder and decoder are random layers; the measurement produces a
    classical bit string that serves as the compressed representation.
    """
    def __init__(self, input_qubits: int = 4, latent_qubits: int = 2, n_ops: int = 8) -> None:
        super().__init__()
        self.input_qubits = input_qubits
        self.latent_qubits = latent_qubits
        self.encoder = tq.RandomLayer(n_ops=n_ops, wires=list(range(input_qubits)))
        self.decoder = tq.RandomLayer(n_ops=n_ops, wires=list(range(latent_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor of shape (batch, input_qubits)
        Returns a compressed tensor of shape (batch, latent_qubits)
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.input_qubits, bsz=bsz, device=device)
        self.encoder(qdev, x)
        measurement = self.measure(qdev)
        # Only keep the first `latent_qubits` bits
        return measurement[:, : self.latent_qubits]


# --------------------------------------------------------------------------- #
# Variational quanvolution filter
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(tq.QuantumModule):
    """
    Variational filter that applies a random circuit to each 2×2 patch.

    The circuit encodes the patch into a 4‑qubit state, applies a random layer,
    and measures all qubits. The measurement vector is returned for each patch.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor of shape (batch, 1, 28, 28)
        Returns a tensor of shape (batch, num_patches, n_wires)
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches: list[torch.Tensor] = []

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
        return torch.cat(patches, dim=1)  # (batch, 196, 4)


# --------------------------------------------------------------------------- #
# Quantum classifier with per‑patch autoencoding
# --------------------------------------------------------------------------- #

class QuanvolutionQuantumAutoencoderClassifier(tq.QuantumModule):
    """
    Quantum classifier that compresses each patch with a quantum autoencoder,
    aggregates the compressed representations, and uses a classical linear head.
    """
    def __init__(
        self,
        num_classes: int = 10,
        latent_qubits: int = 2,
        n_ops: int = 8,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.patch_size = 2
        self.num_patches = (28 // self.patch_size) ** 2
        self.quantum_filter = QuanvolutionFilter(n_wires=self.patch_size * self.patch_size, n_ops=n_ops)
        self.autoencoder = QuantumAutoencoder(input_qubits=self.patch_size * self.patch_size,
                                             latent_qubits=latent_qubits,
                                             n_ops=n_ops)
        self.classifier = nn.Linear(latent_qubits, num_classes)
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        x : torch.Tensor of shape (batch, 1, 28, 28)
        Returns log‑softmax logits of shape (batch, num_classes).
        """
        # Step 1: per‑patch measurement
        patch_meas = self.quantum_filter(x)  # (batch, 196, 4)

        # Step 2: per‑patch quantum autoencoding
        latent_vectors = []
        for i in range(patch_meas.size(1)):
            patch_vec = patch_meas[:, i, :]  # (batch, 4)
            latent = self.autoencoder(patch_vec)  # (batch, latent_qubits)
            latent_vectors.append(latent)
        latent_stack = torch.stack(latent_vectors, dim=1)  # (batch, 196, latent_qubits)

        # Step 3: aggregate compressed patches
        aggregated = latent_stack.mean(dim=1)  # (batch, latent_qubits)

        # Step 4: classical linear head
        logits = self.classifier(aggregated)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionQuantumAutoencoderClassifier"]
