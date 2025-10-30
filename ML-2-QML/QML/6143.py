"""QuanvolutionHybrid – quantum version using a variational kernel on 2×2 patches."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumPatchEncoder(tq.QuantumModule):
    """Encodes a 2×2 patch into a 4‑qubit state with a trainable variational circuit."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layers = [
            tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires))) for _ in range(self.n_layers)
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, patch)
        for layer in self.var_layers:
            layer(qdev)
        return self.measure(qdev).view(-1)


class QuanvolutionHybridClassifier(tq.QuantumModule):
    """
    Quantum hybrid network: a 2×2 patch is encoded with ``QuantumPatchEncoder``,
    the resulting measurement vector is flattened and passed through a classical
    linear classifier. The module is compatible with TorchQuantum's
    ``.forward`` signature.
    """

    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.qencoder = QuantumPatchEncoder(n_wires, n_layers)
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.qencoder.n_wires, bsz=bsz, device=device)

        # Extract 2×2 patches
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
                patches.append(self.qencoder(qdev, patch))

        # Concatenate all patch measurements
        patch_vecs = torch.cat(patches, dim=1)  # (B, 4*14*14)
        logits = self.fc(patch_vecs)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
