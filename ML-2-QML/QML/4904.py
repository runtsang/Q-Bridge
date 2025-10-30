"""Quantum‑variational counterpart of the hybrid quanvolution‑autoencoder.

The quantum model mirrors the classical design but replaces the
classical convolution and autoencoder with a variational circuit that
acts on 2×2 image patches.  The circuit uses a RandomLayer encoder,
a lightweight RealAmplitudes ansatz (acting as a quantum autoencoder),
and a sampler‑style measurement that feeds into a classical linear
classifier.  The implementation relies on the `torchquantum` package
and is fully compatible with PyTorch tensors on GPU or CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Quantum filter – 4‑qubit RandomLayer encoder
# --------------------------------------------------------------------------- #
class QuantumFilter(tq.QuantumModule):
    """Extract 2×2 patches and encode them with a RandomLayer."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Simple 4‑qubit encoder: each pixel → ry gate
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> torch.Tensor:
        """Encode a batch of 4‑dim input vectors into the device."""
        self.encoder(qdev, data)          # data shape [batch, 4]
        self.random_layer(qdev)           # inject randomness
        return qdev


# --------------------------------------------------------------------------- #
# 2. Quantum autoencoder – RealAmplitudes ansatz
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(tq.QuantumModule):
    """Variational circuit that serves as a quantum autoencoder."""

    def __init__(self, latent_dim: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_wires = latent_dim + 2  # add 2 trash qubits
        self.ansatz = tq.RealAmplitudes(
            n_qubits=self.n_wires, reps=3, entanglement="full"
        )

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Apply the ansatz on the current device."""
        self.ansatz(qdev)
        return qdev


# --------------------------------------------------------------------------- #
# 3. Sampler‑style QNN – simple parameterised circuit
# --------------------------------------------------------------------------- #
class SamplerQNNQuantum(tq.QuantumModule):
    """A lightweight parameterised circuit used as a sampler QNN."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.ansatz = tq.RealAmplitudes(n_qubits=self.n_wires, reps=2)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.ansatz(qdev)
        return qdev


# --------------------------------------------------------------------------- #
# 4. Hybrid quantum classifier
# --------------------------------------------------------------------------- #
class QuanvolutionAutoencoderQNN(tq.QuantumModule):
    """Full quantum hybrid model.

    Forward pass:
        1. Extract 2×2 patches → QuantumFilter (RandomLayer encoder).
        2. Apply QuantumAutoencoder ansatz on each patch.
        3. Measure all qubits → 4‑dim feature vector per patch.
        4. Concatenate all patch vectors → 784‑dim feature vector.
        5. Pass through a classical linear classifier to output logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.filter = QuantumFilter()
        self.autoencoder = QuantumAutoencoder()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head maps 784 → 10 classes
        self.linear = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)  # MNIST shape
        patches = []

        # Iterate over 2×2 patches
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Gather four pixel values per sample
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # shape [N, 4]

                # Quantum device for this patch
                qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=device)
                # Encode, random layer, autoencoder ansatz
                self.filter(qdev, data)
                self.autoencoder(qdev)
                # Measurement → 4‑dim output per sample
                meas = self.measure(qdev).view(bsz, 4)
                patches.append(meas)

        # Concatenate all patch measurements
        features = torch.cat(patches, dim=1)  # shape [N, 784]
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumFilter", "QuantumAutoencoder", "SamplerQNNQuantum", "QuanvolutionAutoencoderQNN"]
