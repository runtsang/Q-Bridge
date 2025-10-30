"""Hybrid quanvolutional network – quantum implementation.

The quantum version replaces the classical convolutional patch extraction
with a small variational circuit that processes 2×2 image patches.
Each patch is encoded by a four‑wire quantum device, run through
a randomized layer, and measured in the Pauli‑Z basis.  The resulting
four‑bit feature vector is concatenated across the image and passed
through a linear classifier.  The architecture is fully differentiable
and can be trained end‑to‑end with gradient‑based optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum‑enhanced quanvolutional network.

    The class inherits from :class:`tq.QuantumModule` and implements
    a patch‑wise variational circuit.  The encoder uses a GeneralEncoder
    that applies Ry rotations to each input pixel.  The variational
    layer consists of a RandomLayer followed by a sequence of
    trainable RX gates and a fixed CNOT pattern, producing a 4‑dimensional
    feature vector for each patch.  After concatenation the features
    are fed into a linear head that maps to the final logits.
    """

    class _QLayer(tq.QuantumModule):
        """Variational block applied to every patch."""

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
            self.random_layer = tq.RandomLayer(
                n_ops=10, wires=list(range(self.n_wires))
            )
            self.rx_params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=x.shape[0],
                device=x.device,
            )
            self.encoder(qdev, x)
            self.random_layer(qdev)
            for wire, gate in enumerate(self.rx_params):
                gate(qdev, wires=wire)
            # Fixed CNOT chain to entangle qubits
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(
        self,
        out_features: int = 10,
    ) -> None:
        super().__init__()
        self.qlayer = self._QLayer()
        self.linear = nn.Linear(4 * 14 * 14, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log‑softmax logits for a batch of images."""
        bsz = x.shape[0]
        # reshape 28×28 image into 2×2 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                patches.append(patch)
        # apply quantum block to each patch
        outputs = [self.qlayer(p) for p in patches]
        # concatenate all patch outputs
        features = torch.cat(outputs, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
