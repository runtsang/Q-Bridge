"""Quantum quanvolutional filter with a trainable parameterized ansatz and adjustable depth."""

import torch
import torch.nn as nn
import torchquantum as tq


class ParameterizedQuantumFilter(tq.QuantumModule):
    """Parameterized quantum kernel for 2×2 image patches."""
    def __init__(self, depth: int = 2):
        super().__init__()
        self.n_wires = 4
        self.depth = depth

        # Encoder maps classical pixel values to rotation angles
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Trainable parameters: depth × n_wires × 3 (for Rz, Ry, Rz)
        self.params = nn.Parameter(torch.randn(depth, self.n_wires, 3))

        self.measure = tq.MeasureAll(tq.PauliZ)

    def _apply_ansatz(self, qdev: tq.QuantumDevice, layer_idx: int) -> None:
        """Apply a single layer of parameterized rotations."""
        for wire in range(self.n_wires):
            rz1 = self.params[layer_idx, wire, 0]
            ry = self.params[layer_idx, wire, 1]
            rz2 = self.params[layer_idx, wire, 2]
            qdev.rz(rz1, wire)
            qdev.ry(ry, wire)
            qdev.rz(rz2, wire)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        x = x.view(bsz, 28, 28)
        patches = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Gather 2×2 pixel patch
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)

                # Apply depth‑controlled ansatz
                for d in range(self.depth):
                    self._apply_ansatz(qdev, d)

                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier with the parameterized quantum filter."""
    def __init__(self, depth: int = 2):
        super().__init__()
        self.qfilter = ParameterizedQuantumFilter(depth=depth)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return torch.nn.functional.log_softmax(logits, dim=-1)


__all__ = ["ParameterizedQuantumFilter", "QuanvolutionClassifier"]
