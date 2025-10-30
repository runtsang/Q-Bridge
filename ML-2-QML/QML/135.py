"""Quantum quanvolution filter with variational circuit.

This module implements a quantum filter that processes 2x2 patches of an image
with a trainable variational circuit. The circuit is a stack of
parameterised rotation gates and CNOT entanglement gates. The output of the
measurements is concatenated across patches and fed into a classical
classifier head.

The implementation is compatible with torchquantum and can be run on CPU or
GPU backends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class Quanvolution__gen148(nn.Module):
    """
    Hybrid quantum-classical network.

    Parameters
    ----------
    patch_size : int, default 2
        Size of the square patch to process.
    n_qubits : int, default 4
        Number of qubits used for each patch (must be >= patch_size**2).
    n_layers : int, default 2
        Number of variational layers applied to each patch.
    dropout_prob : float, default 0.0
        Dropout probability applied to the concatenated quantum features.
    """
    def __init__(
        self,
        patch_size: int = 2,
        n_qubits: int = 4,
        n_layers: int = 2,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if n_qubits < patch_size ** 2:
            raise ValueError("n_qubits must be at least patch_size**2")
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

        # Encoder: map each pixel to a rotation around Y
        self.encoder = tq.GeneralEncoder(
            [
                {
                    "input_idx": [i],
                    "func": "ry",
                    "wires": [i],
                }
                for i in range(patch_size ** 2)
            ]
        )

        # Variational layer: repeated layers of rotations and entanglement
        self.var_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.var_layers.append(
                tq.Layer(
                    n_ops=2 * n_qubits,
                    wires=list(range(n_qubits)),
                    op_types=["ry", "rz"],
                )
            )

        # Entanglement pattern: CNOT chain
        self.entanglement = tq.CNOT(list(range(n_qubits - 1)), wires=list(range(n_qubits - 1)))

        # Measurement: all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Linear head
        self.feature_dim = n_qubits * (28 // patch_size) ** 2
        self.classifier = nn.Linear(self.feature_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log-probabilities over 10 classes.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)

        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                # Extract patch and flatten
                patch = x[:, r : r + self.patch_size, c : c + self.patch_size]
                patch = patch.view(bsz, -1)
                # Encode
                self.encoder(qdev, patch)
                # Apply variational layers
                for layer in self.var_layers:
                    layer(qdev)
                    self.entanglement(qdev)
                # Measure
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, self.n_qubits))

        # Concatenate all patch features
        features = torch.cat(patches, dim=1)
        features = self.dropout(features)

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the concatenated quantum feature vector before the classifier.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)

        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = x[:, r : r + self.patch_size, c : c + self.patch_size]
                patch = patch.view(bsz, -1)
                self.encoder(qdev, patch)
                for layer in self.var_layers:
                    layer(qdev)
                    self.entanglement(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, self.n_qubits))

        features = torch.cat(patches, dim=1)
        return features


__all__ = ["Quanvolution__gen148"]
