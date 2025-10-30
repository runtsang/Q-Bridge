"""Quanvolution filter with trainable variational quantum circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """
    Quantum filter that encodes 2×2 image patches into a
    4‑qubit system, applies a two‑layer Ry‑CNOT ansatz, and
    measures all qubits in the computational basis.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder: map pixel intensities to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable parameters for a 2‑layer Ry–CNOT ansatz
        self.theta = nn.Parameter(torch.randn(2, self.n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _ansatz(self, qdev: tq.QuantumDevice) -> None:
        # First layer of Ry gates
        for i in range(self.n_wires):
            qdev.ry(self.theta[0, i], i)
        # CNOT chain
        for i in range(self.n_wires - 1):
            qdev.cnot(i, i + 1)
        # Second layer of Ry gates
        for i in range(self.n_wires):
            qdev.ry(self.theta[1, i], i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
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
                self._ansatz(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network that uses the quantum filter followed by a
    classical linear classifier.
    """

    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
