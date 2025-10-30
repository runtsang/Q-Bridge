"""Quantum quanvolutional network with a parameter‑tuned variational circuit.

The network applies a 2‑qubit variational layer to each 2×2 image patch
and follows it with a linear head.  A lightweight classical shortcut
is available for quick baseline experiments by setting ``trainable=False``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionNet(tq.QuantumModule):
    """Hybrid quantum‑classical quanvolutional network.

    Parameters
    ----------
    trainable : bool, optional
        If ``True`` the quantum kernel is used; otherwise a classical
        convolutional approximation is employed.  Default is ``True``.
    """
    def __init__(self, trainable: bool = True) -> None:
        super().__init__()
        self.trainable = trainable

        if trainable:
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.var_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
        else:
            # Classical shortcut: depthwise separable 2×2 conv
            self.classical_conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            bsz, _, _, _ = x.shape
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

            patches = []
            x_flat = x.view(bsz, 28, 28)
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = torch.stack(
                        [
                            x_flat[:, r, c],
                            x_flat[:, r, c + 1],
                            x_flat[:, r + 1, c],
                            x_flat[:, r + 1, c + 1],
                        ],
                        dim=1,
                    )
                    self.encoder(qdev, patch)
                    self.var_layer(qdev)
                    measurement = self.measure(qdev)
                    patches.append(measurement.view(bsz, 4))
            features = torch.cat(patches, dim=1)
        else:
            features = self.classical_conv(x)
            features = features.view(x.size(0), -1)

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)
