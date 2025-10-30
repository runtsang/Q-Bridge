"""
Quantum quanvolution‑classifier with a variational self‑attention module.

The quantum variant shares the same public API as the classical one but
replaces the convolution and attention with parameterised quantum circuits.
The circuits are implemented via TorchQuantum; they are entirely
device‑agnostic and can be run on CPU or GPU simulators.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.quantum as tq  # TorchQuantum
import torch.nn.functional as F


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum analogue of the 2×2 convolution.  For each 2×2 patch we
    encode the four pixel values into four qubits, apply a random
    variational layer, and measure all qubits.  The measurement
    outcomes form the feature vector for that patch.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # reshape to 28×28 images
        x_img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x_img[:, r, c],
                        x_img[:, r, c + 1],
                        x_img[:, r + 1, c],
                        x_img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumSelfAttention(tq.QuantumModule):
    """
    Variational self‑attention circuit.  For a batch of patch features it
    produces a scalar weight for each patch by encoding the feature,
    running a small ansatz, and measuring the expectation of Z on
    the first qubit.  The raw expectation values are passed through a
    softmax to obtain normalized attention weights.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.circuit = tq.Circuit(
            tq.CX(0, 1),
            tq.CX(1, 2),
            tq.CX(2, 3),
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, num_patches, 4)

        Returns
        -------
        Tensor
            Attention weights of shape (batch, num_patches)
        """
        bsz, n_patches, _ = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)

        raw_weights = []
        for i in range(n_patches):
            # encode patch
            self.encoder(qdev, x[:, i, :])
            # apply ansatz
            self.circuit(qdev)
            # measure expectation of Z on qubit 0
            meas = self.measure(qdev)
            # take first qubit expectation as weight
            weight = meas[:, 0].float()
            raw_weights.append(weight.unsqueeze(-1))

        raw_weights = torch.cat(raw_weights, dim=1)  # (batch, n_patches)
        # softmax over patches
        return torch.softmax(raw_weights, dim=1)


class QuantumQuanvolutionClassifier(nn.Module):
    """
    End‑to‑end quantum classifier.  A quanvolution filter extracts
    patch features, a variational self‑attention module re‑weights
    them, and a linear head produces class logits.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.attention = QuantumSelfAttention()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Obtain raw patch features
        features = self.qfilter(x)  # (batch, 4*14*14)
        # Reshape to (batch, num_patches, feature_dim)
        patches = features.view(-1, 14 * 14, 4)  # (batch, 196, 4)

        # Compute attention weights
        attn_weights = self.attention(patches)  # (batch, 196)

        # Apply weights to flattened features
        weighted_features = patches.view(-1, 4 * 14 * 14) * attn_weights.unsqueeze(-1)

        logits = self.linear(weighted_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuantumSelfAttention", "QuantumQuanvolutionClassifier"]
