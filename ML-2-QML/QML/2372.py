"""Hybrid quantum quanvolution filter with random circuit and thresholding.

This module implements a quantum filter that can be used as a drop‑in
replacement for the classical version.  It applies a random variational
circuit to each 2×2 patch of the input image, measures all qubits in the
Pauli‑Z basis and returns the mean probability of measuring |1>.  An
optional threshold can be supplied to binarise the output.

Classes
-------
QuanvolutionFilter : tq.QuantumModule
    Quantum filter that processes image patches.
QuanvolutionClassifier : nn.Module
    Simple classifier that stacks the filter and a linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a random circuit to 2×2 image patches.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used per patch (2×2 grid).
    n_ops : int, default 8
        Number of random two‑qubit gates in the variational layer.
    threshold : float, default 0.0
        Threshold applied to the mean measurement probability.
    """

    def __init__(self, n_qubits: int = 4, n_ops: int = 8, threshold: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.threshold = threshold

        # Encoder: map classical pixel values to Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_qubits)
            ]
        )
        # Variational layer with random two‑qubit gates
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_qubits)))
        # Measurement of all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W) with single channel.

        Returns
        -------
        torch.Tensor
            Concatenated measurement results for all patches, shape
            (B, n_qubits * H' * W').
        """
        bsz = x.shape[0]
        device = x.device

        # Build a quantum device with batch support
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=device)

        # Reshape to (B, H, W)
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2×2 patch and flatten
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode classical data into Ry rotations
                self.encoder(qdev, data)
                # Apply random variational layer
                self.q_layer(qdev)
                # Measure all qubits
                measurement = self.measure(qdev)
                # measurement is a tensor shape (B, n_qubits)
                # Optionally apply threshold to binarise
                if self.threshold!= 0.0:
                    measurement = torch.where(
                        measurement > self.threshold, torch.ones_like(measurement), torch.zeros_like(measurement)
                    )
                patches.append(measurement.view(bsz, self.n_qubits))

        # Concatenate all patches along feature dimension
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quantum filter followed by a linear head.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits per patch.
    n_ops : int, default 8
        Number of random two‑qubit gates.
    threshold : float, default 0.0
        Threshold applied to the measurement probabilities.
    num_classes : int, default 10
        Number of target classes.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_ops: int = 8,
        threshold: float = 0.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(
            n_qubits=n_qubits, n_ops=n_ops, threshold=threshold
        )
        # Compute flattened feature size
        dummy = torch.zeros(1, 1, 28, 28)
        feat = self.qfilter(dummy)
        self.linear = nn.Linear(feat.shape[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
