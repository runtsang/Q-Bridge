"""Hybrid quantum model combining a quanvolution filter and a quantum fully‑connected layer.

The model first maps each 2×2 image patch through a 4‑qubit quantum kernel
(ry gates + random layer).  The resulting 4‑dimensional feature per patch
is averaged across the image to obtain a 4‑dimensional vector that is
encoded into a 4‑wire quantum device.  A second random layer is applied,
after which the Pauli‑Z measurement produces the final 4‑dimensional
output, normalised with BatchNorm1d.

This architecture merges the patch‑wise quantum feature extraction
from the quanvolution example with the fully‑connected quantum layer
from the Quantum‑NAT example, providing a deeper quantum representation
while keeping the overall circuit depth modest.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumPatchFilter(tq.QuantumModule):
    """Apply a 4‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model inspired by the Quantum‑NAT paper."""
    def __init__(self):
        super().__init__()
        self.patch_filter = QuantumPatchFilter()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
        patches = self.patch_filter(x)  # (bsz, 4 * 14 * 14)
        patches = patches.view(bsz, 4, -1).mean(dim=2)  # (bsz, 4)
        self.encoder(qdev, patches)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
